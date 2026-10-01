import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from touhou_ai.telemetry_memory import TelemetryMemory, CAPACITY


@unittest.skipUnless(os.name == 'nt', 'Windows named shared memory')
class MemoryTests(unittest.TestCase):
    def test_cross_process_and_reader_reconnect_without_disk(self):
        with tempfile.TemporaryDirectory() as root:
            reader = TelemetryMemory(root)
            try:
                self.assertIsNone(reader.read())
                code = "from touhou_ai.telemetry_memory import TelemetryMemory; import sys; c=TelemetryMemory(sys.argv[1]); assert c.publish({'episode_id':'one','value':7}); c.close()"
                subprocess.run([sys.executable, '-c', code, root], check=True)
                sequence, payload = reader.read()
                self.assertEqual(json.loads(payload)['value'], 7)
                self.assertIsNone(reader.read(sequence))
                second = TelemetryMemory(root)
                try:
                    self.assertEqual(second.read(), (sequence, payload))
                    self.assertTrue(second.publish({'episode_id': 'two', 'value': 0}))
                    self.assertEqual(json.loads(reader.read()[1])['value'], 0)
                    self.assertFalse(second.publish({'large':'x'*CAPACITY}))
                    self.assertEqual(json.loads(reader.read()[1])['value'], 0)
                finally:
                    second.close()
                self.assertEqual(list(Path(root).iterdir()), [])
            finally:
                reader.close()

    def test_contention_drops_instead_of_blocking_and_never_partial(self):
        with tempfile.TemporaryDirectory() as root:
            channel = TelemetryMemory(root)
            try:
                channel.publish({'old': True})
                self.assertTrue(channel._acquire())
                results=[]
                def attempt():
                    results.append(channel.publish({'new': True}))
                    results.append(channel.read())
                t=threading.Thread(target=attempt)
                t.start();t.join(2)
                self.assertFalse(t.is_alive())
                self.assertEqual(results,[False,None])
                channel.kernel.ReleaseMutex(channel.mutex)
                self.assertEqual(json.loads(channel.read()[1]), {'old':True})
            finally:
                channel.close()

    def test_http_and_sse_use_memory_not_legacy_file(self):
        import http.client
        from touhou_ai.dashboard import make_server
        from touhou_ai.telemetry_memory import publish
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); artifacts=root/'artifacts'
            (artifacts/'telemetry').mkdir(parents=True)
            (artifacts/'telemetry/latest.json').write_text('{"obsolete":true}')
            server=make_server(root,0)
            thread=threading.Thread(target=server.serve_forever);thread.start()
            try:
                self.assertTrue(publish({'memory':True},artifacts))
                for endpoint in ('/api/live','/api/live-stream'):
                    client=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
                    try:
                        with patch.object(Path,'read_text',side_effect=AssertionError('file read')):
                            client.request('GET',endpoint);response=client.getresponse()
                            self.assertEqual(response.status,200)
                            body=response.readline() if endpoint.endswith('stream') else response.read()
                            if endpoint.endswith('stream'):body=body.removeprefix(b'data: ')
                            self.assertEqual(json.loads(body),{'memory':True})
                    finally:client.close()
            finally:
                server.shutdown();thread.join();server.server_close()

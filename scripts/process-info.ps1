if (-not ('TouhouProcessInfo' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class TouhouProcessInfo {
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern IntPtr OpenProcess(uint access, bool inherit, int id);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern bool QueryFullProcessImageName(IntPtr process, uint flags, StringBuilder name, ref int size);
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr handle);
    [DllImport("advapi32.dll", SetLastError=true)]
    static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)]
    static extern bool GetTokenInformation(IntPtr token, int type, out int value, int size, out int needed);
    public static bool IsElevated(int id) {
        IntPtr process = OpenProcess(0x1000, false, id);
        if (process == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        IntPtr token = IntPtr.Zero;
        try {
            if (!OpenProcessToken(process, 8, out token)) throw new Win32Exception(Marshal.GetLastWin32Error());
            int value, needed;
            if (!GetTokenInformation(token, 20, out value, 4, out needed))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return value != 0;
        } finally { if (token != IntPtr.Zero) CloseHandle(token); CloseHandle(process); }
    }
    public static string ImagePath(int id) {
        IntPtr handle = OpenProcess(0x1000, false, id);
        if (handle == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        try {
            int size = 32768;
            var name = new StringBuilder(size);
            if (!QueryFullProcessImageName(handle, 0, name, ref size))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return name.ToString();
        } finally { CloseHandle(handle); }
    }
}
'@
}

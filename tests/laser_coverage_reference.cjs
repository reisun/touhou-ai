function gridLaserCoverage(entities, origin, cell, width, height) {
 const coverage=new Float32Array(width*height);
 for(const e of entities||[]) {
  const g=e.collision;if(!g?.active)continue;
  const px=g.origin[0]+192,py=g.origin[1],co=Math.cos(g.angle),si=Math.sin(g.angle);
  const corners=[];for(const t of [0,g.length])for(const w of [-g.width/2,g.width/2])corners.push([px+co*t-si*w,py+si*t+co*w]);
  const x0=Math.max(0,Math.floor((Math.min(...corners.map(p=>p[0]))-origin[0])/cell));
  const x1=Math.min(width,Math.ceil((Math.max(...corners.map(p=>p[0]))-origin[0])/cell));
  const y0=Math.max(0,Math.floor((Math.min(...corners.map(p=>p[1]))-origin[1])/cell));
  const y1=Math.min(height,Math.ceil((Math.max(...corners.map(p=>p[1]))-origin[1])/cell));
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++) {
   let area=0;for(const sx of [.25,.75])for(const sy of [.25,.75]) {
    const dx=origin[0]+(x+sx)*cell-px,dy=origin[1]+(y+sy)*cell-py;
    const along=dx*co+dy*si,across=-dx*si+dy*co;
    if(along>=0&&along<=g.length&&Math.abs(across)<=g.width/2)area+=.25;
   }
   coverage[y*width+x]=Math.max(coverage[y*width+x],area);
  }
 }
 return coverage;
}

module.exports=gridLaserCoverage;

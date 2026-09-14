"""Build mmap-friendly decoded image cache.

Example: python scripts/data/build_video_cache.py DATASET CACHE --max-frames 2000
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(); p.add_argument('dataset'); p.add_argument('cache')
    p.add_argument('--max-frames', type=int, default=None); p.add_argument('--size', type=int, default=224)
    p.add_argument('--sources', nargs='+', default=None, help='Video feature names to cache')
    a=p.parse_args(); os.makedirs(a.cache, exist_ok=True)
    info=json.load(open(os.path.join(a.dataset,'meta','info.json')))
    total=int(info['total_frames']); n=min(total,a.max_frames) if a.max_frames else total
    sources=[k for k,v in info['features'].items() if v.get('dtype')=='video']
    if a.sources:
        missing=set(a.sources)-set(sources)
        if missing: p.error(f'unknown video sources: {sorted(missing)}')
        sources=a.sources
    for source in sources:
        paths=sorted(glob.glob(os.path.join(a.dataset,'videos',source,'chunk-*','*.mp4')))
        out=os.path.join(a.cache,source.replace('/','__')+'.npy')
        partial=out+'.partial'
        arr=np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8, shape=(n,a.size,a.size,3))
        import av; written=0
        for path in paths:
            c=av.open(path); stream=c.streams.video[0]
            for frame in c.decode(stream):
                if written>=n: break
                im=Image.fromarray(frame.to_ndarray(format='rgb24'))
                input_w,input_h=im.size; scale=max(a.size/input_w,a.size/input_h)
                resized_w=int(np.ceil(input_w*scale)); resized_h=int(np.ceil(input_h*scale))
                im=im.resize((resized_w,resized_h),Image.Resampling.BILINEAR)
                left=(resized_w-a.size)//2; top=(resized_h-a.size)//2
                im=im.crop((left,top,left+a.size,top+a.size))
                arr[written]=np.asarray(im,dtype=np.uint8); written+=1
            c.close()
            if written>=n: break
        if written!=n: raise RuntimeError(f'{source}: decoded {written}, expected {n}')
        arr.flush(); del arr; os.replace(partial,out)
        print(f'{source}: {out} ({os.path.getsize(out)/1e6:.1f} MB)',flush=True)
    for source in sources:
        json.dump({'frames':n,'size':a.size,'source':source},open(os.path.join(a.cache,source.replace('/','__')+'.json'),'w'),indent=2)
if __name__=='__main__': main()

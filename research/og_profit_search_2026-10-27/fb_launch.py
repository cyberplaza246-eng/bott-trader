import subprocess, sys, os
os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',CUDA_VISIBLE_DEVICES='-1')
H=os.path.dirname(os.path.abspath(__file__))
N=138513; CH=3000; W=int(sys.argv[1])
chunks=[(s,min(N,s+CH)) for s in range(0,N,CH)][::-1]   # newest first
procs=[]
for w in range(W):
    mine=chunks[w::W]
    arg=','.join(f'{a}:{b}' for a,b in mine)
    p=subprocess.Popen([sys.executable,os.path.join(H,'code','fb_worker2.py'),arg,os.path.join(H,'out',f'fb_n{w}.jsonl'),os.path.join(H,'out',f'fb_n{w}.prog')],
                       stdout=open(os.path.join(H,'out',f'fbn{w}.out'),'w'),stderr=open(os.path.join(H,'out',f'fbn{w}.err'),'w'),creationflags=0x00004000)  # BELOW_NORMAL
    procs.append(p); print(w,p.pid,arg[:60],flush=True)
for p in procs: p.wait()

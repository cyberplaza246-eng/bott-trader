s=open('code/fb_worker.py',encoding='utf-8').read().replace('\r\n','\n')
# signature: main(ranges, outpath, progress)
s=s.replace("def main(start_i, end_i, outpath, progress):","def main(ranges, outpath, progress):\n    start_i=ranges[0][0]; end_i=ranges[-1][1]")
# reduced models toggle
old="            rec['sweep']=sd\n            r=et.get_trading_signal(dfe,'MNQ')"
new="            rec['sweep']=sd\n            et.lstm_available=et.intelligent_available=et.advanced_strats_available=(sd in ('BUY','SELL'))\n            r=et.get_trading_signal(dfe,'MNQ')"
assert old in s; s=s.replace(old,new)
# loop over ranges
old2="    for i in range(max(start_i,99), min(end_i,len(b))):\n"
assert old2 in s
i0=s.index(old2); i1=s.index("    open(progress,'w').write(json.dumps({'start':start_i,'end':end_i,'done':n,'FINISHED'")
body=s[i0:i1]
body_ind=''.join('    '+l if l.strip() else l for l in body.split('\n')[1:])  # body of loop indented one more level
newloop="    for (rs,re_) in ranges:\n      for i in range(max(rs,99), min(re_,len(b))):\n"
# the original loop body lines are indented 8 spaces; need them at 8 spaces under 'for i' at 6 -> they are deeper than 6, OK as is
body_lines=body.split('\n')[1:]
s=s[:i0]+newloop+'\n'.join(body_lines)+s[i1:]
s=s.replace("if __name__=='__main__':\n    main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])","if __name__=='__main__':\n    rng=[tuple(int(x) for x in p.split(':')) for p in sys.argv[1].split(',')]\n    main(rng,sys.argv[2],sys.argv[3])")
open('code/fb_worker2.py','w',encoding='utf-8').write(s)
import ast; ast.parse(s); print('ok')

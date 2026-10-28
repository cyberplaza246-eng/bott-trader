import sim,time
t=time.time(); C=sim.load_signals(); F=sim.load_fine(); print(len(C),len(F),time.time()-t)
df,o=sim.simulate(C,F); print(len(df),o); print(sim.metrics(df)); print(df.head(5).to_string())

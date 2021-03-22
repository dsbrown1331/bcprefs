import pickle
demo_files = ['demonstrations/cartpole20.p','demonstrations/cartpole5.p', 'demonstrations/cartpole2.p']
for dfile in demo_files:
    demos = pickle.load( open( dfile, "rb" ) )
    print("loaded demos", dfile, len(demos))
    dlengths = []
    for s,a in demos:
        dlengths.append(len(s))
    print(dlengths)

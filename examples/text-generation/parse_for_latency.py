import sys

def process(log):
    with open(log) as f:
        warmup_done = False
        t0 = None
        list_of_latencies = []
        for ln in f.readlines():
            if 'Running generate...' in ln:
                warmup_done = True
            if warmup_done:
                if 'First Token time(greedy):' in ln and t0 is not None:
                    t1 = float(ln.split(':')[-1])
                    print(t1-t0)
                    list_of_latencies += [t1 - t0]
                    t0=None
                if 'Step4+ starting time is' in ln:
                    t0 = float(ln.split('is')[-1])
        print('Average latency:', sum(list_of_latencies)/len(list_of_latencies))

process(sys.argv[1])
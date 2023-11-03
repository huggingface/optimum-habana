

def process(logfile):
    start_output_collection=''
    started_gen = False
    res = {}
    with open(logfile) as f:
        for ln in f:
            if 'Input/outputs:' in ln:
                splitpt = [i for i,ch in enumerate(start_output_collection) if ch=='['][2]
                nums = ''.join([i for i in start_output_collection[0:splitpt] if i!='\n']).strip(' ').strip(')')
                sent = start_output_collection[splitpt:]
                #print(currshape, start, starttkn, e2e, gc, eval(nums), eval(sent))
                res[currshape] = [start, starttkn, e2e, gc, eval(nums), eval(sent)]
                break
            if len(start_output_collection) > 0:
                if 'Generating for shape' in ln:
                    splitpt = [i for i,ch in enumerate(start_output_collection) if ch=='['][2]
                    nums = ''.join([i for i in start_output_collection[0:splitpt] if i!='\n']).strip(' ').strip(')')
                    sent = start_output_collection[splitpt:]
                    #print(currshape, start, starttkn, e2e, gc, eval(nums), eval(sent))
                    res[currshape] = [start, starttkn, e2e, gc, eval(nums), eval(sent)]
                else:
                    start_output_collection += ln
            if 'Generating for shape' in ln:
                started_gen = True
                start_output_collection = ''
                currshape = int(ln.split(', ')[-1])
            if 'starting time is' in ln and started_gen:
                start = float(ln.split(' ')[-1])
            if 'First Token time' in ln and started_gen:
                starttkn = float(ln.split(':')[-1])
            if 'Total E2E time of this iteration is duration' in ln and started_gen:
                e2e = float(ln.split('=')[-1])
            if 'GC stats' in ln and started_gen:
                gc = eval(ln.split('GC stats ')[-1])
            if 'outputs, size=' in ln and started_gen:
                start_output_collection += ln.split(': tensor(')[-1]
            
    return res

def print_perf(res):
    for shp in res:
        start, starttkn, e2e, gc, nums, sent = res[shp]
        print(shp, starttkn-start, e2e)
    print('-----')

def compare(r0, r1):
    for shp in r0:
        v0 = r0[shp][4]
        v1 = r1[shp][4]
        for i,j in zip(v0,v1):
            mn = min(len(i), len(j))
            if not i[:mn] == j[:mn]:
                print('mismatch', shp)


bk20_reusecache = process('bf16_bkt20_shp15_25_17_30_40_reusecache_hpugraph.txt')
print_perf(bk20_reusecache)
bkNone_noreusecache = process('bf16_bktNone_shp15_25_17_30_40_noreusecache_hpugraph.txt')
print_perf(bkNone_noreusecache)
bkNone_reusecache = process('bf16_bktNone_shp15_25_17_30_40_reusecache_hpugraph.txt')
print_perf(bkNone_reusecache)
compare(bkNone_noreusecache, bkNone_reusecache)
compare(bkNone_noreusecache, bk20_reusecache)
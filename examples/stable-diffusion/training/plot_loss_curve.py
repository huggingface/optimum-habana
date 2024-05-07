import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def sample(x):
    return x//500

def test():
    def match(x,y):
        return np.all(np.abs(np.array(x)-np.array(y)) < 0.00001)
    a = [0.1,0.2,0.3,0.6]
    assert a == fix(a)
    assert match([0.1,0.15,0.2,0.2], fix([0.1,0.1,0.2,0.2]))

def smooth(y):
    return savgol_filter(y, 51, 10)
    # return _smooth(y, 51, 3)
    # take every 3 point
    # then smooth over a window of 51

def _smooth(y, box_pts, sample):
    y = y[::sample]
    box = np.ones(box_pts)/box_pts    
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def style(idx):
    return ['g', 'r', 'b'][idx]

def strip_ln(ln):
    return ln[ln.find("Steps"):].split("/")[0].split("|")[-1], \
           ln[ln.find("step_loss"):].split('step_loss=')[-1].split(']')[0]

        # return ln[ln.find("step_loss="):].strip()
    # else:
    #     return ln[ln.find("{'loss"):].strip()

def filter_fn(ln):
    return "step_loss" in ln# and 'epoch' in ln

# there may be multiple epoch due to truncation of decimal when printing
# expand them so there is unique epoch number for each point
def fix(epoch):
    start_idx = 0
    end_idx = 0
    while True:
        start_idx = end_idx
        end_idx = start_idx
        if end_idx >= len(epoch):
            break
        curr_ep = epoch[start_idx]
        while True:
            if end_idx >= len(epoch):
                break
            next_ep = epoch[end_idx]
            if next_ep != curr_ep:
                break
            else:
                end_idx += 1
        if end_idx != len(epoch):
            if end_idx-start_idx > 1:
                assert epoch[end_idx] - epoch[start_idx] < 1, "Truncation will not cause such a big diff"
            increment = (epoch[end_idx] - epoch[start_idx])/(end_idx - start_idx)
            for idx in range(start_idx, end_idx):
                epoch[idx] += (idx - start_idx) * increment
    return epoch

def parse(flnm, smooth_fn=lambda x:x, clip_first=100):
    with open(flnm) as f:
        loss = []
        steps = []
        eval_samples_per_sec = []
        eval_epoch = 0
        prev_step = 0
        last_loss = 0.0
        for ln in f.readlines():
            if not filter_fn(ln):
                continue
            step, step_loss = strip_ln(ln)

            if 'step_loss' in ln:
                if prev_step != int(step):
                    loss.append(last_loss)
                    steps.append(int(prev_step))
                    prev_step = int(step)
                    #print("\nstep/loss", step, last_loss)
                else:
                    last_loss = float(step_loss)
            #TODO: parse eval epoch?
        loss.append(last_loss)
        steps.append(int(prev_step))
    SAMPLE= sample(len(loss))
    loss = (loss[clip_first:])[::SAMPLE]
    steps = (steps[clip_first:])[::SAMPLE]

    loss = smooth_fn(loss)
    #epoch=fix(epoch) #TODO uncomment this
    return steps, loss, eval_epoch, eval_samples_per_sec, flnm.split('/')[-1].split('.')[0]

def plot(infolist, name):
    for idx, (steps, loss, eval_epoch, eval_loss, tag) in enumerate(infolist):
        #assert eval_epoch <= epoch[-1] <= eval_epoch+1
        if len(loss) > 0:
            plt.plot(steps, loss, label=f'Train_{tag}', color=style(idx), marker='.')
        if len(eval_loss) > 0:
            plt.plot(range(eval_epoch), eval_loss, label=f'Eval_{tag}', color=style(idx), marker='o')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    print("Write name ", name)
    plt.title(name + ' train and eval loss vs epoch')
    plt.savefig('loss_plot_' + name + '.png')


def main(filenames, do_smooth, clip_first=100, name='stdxl'):
    if ',' in filenames:
        filenames = filenames.split(',')
    else:
        filenames = [filenames]
    smooth_fn = smooth if do_smooth else lambda x:x
    plot([parse(flnm, smooth_fn, clip_first) for flnm in filenames], name)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], True, 100, name=sys.argv[2])
    else:
        main(sys.argv[1], True, 100, name='stdxl')
    #test()
    # python plot_loss_curve.py log.txt
    # python plot_loss_curve.py log1.txt,log2.txt
    # python plot_loss_curve.py log1.txt,log2.txt Llama1


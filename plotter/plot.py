import matplotlib.animation as ani
from matplotlib.lines import Line2D
import matplotlib.offsetbox as mob
import matplotlib.pyplot as plt
import numpy as np

data_dir = "../data/"
plots_dir = "../plots/"
frame_delay = 200
default_frames = 200
long_frames = 265
short_frames = 195
last_frames = 370

data = [
    (0.0842,0.0270,long_frames),
    (0.260,0.0201,short_frames),
    (0.240,0.0304,default_frames),
    (0.174,0.0325,last_frames),
    (0.287,0.0199,default_frames),
    (0.391,0.0117,default_frames),
]

class Trial:
    def __init__(self, vel=0.0, grad=0.0, frames=200):
        self.vel = vel
        self.grad = grad
        self.frames = frames
        self.tag = "Vp_{}_".format(round(vel, 2))
        snapshot = np.loadtxt(data_dir + self.tag + "c_0.txt", dtype='d')
        self.init_data = thicken(snapshot)
        past = np.transpose(np.loadtxt(data_dir + self.tag + "c_history_temp.txt", dtype='d'))
        last = np.loadtxt(data_dir + self.tag +"c_{}.txt".format(frames), dtype='d')
        combined = np.concatenate((past, last), axis=1)
        self.min = np.min(combined)
        self.max = np.max(combined)
        self.history_data = thicken(combined)
        self.ymax, self.xmax = self.history_data.shape


class AnchoredHScaleBar(mob.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = mob.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        # vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        # vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        # size_bar.add_artist(vline1)
        # size_bar.add_artist(vline2)
        txt = mob.TextArea(label, multilinebaseline=False) # formerly minimumdescent
        self.vpac = mob.VPacker(children=[size_bar,txt],
                                align="center", pad=ppad, sep=sep)
        mob.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                       borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                                       **kwargs)

def main():
    # temp = Trial(vel=0.174, grad=0.0325, frames=last_frames)
    # make_visualizations(temp)
    trials = [Trial(vel, grad, frames) for (vel, grad, frames) in data]
    for trial in trials:
        make_visualizations(trial)
    make_composite_visual(trials)
    return None

def thicken(data,times=3):
    itr = tuple([data for i in range(times)])
    return np.vstack(itr)

def make_composite_visual(trials):
    idx = 0
    height_min = sum([trial.ymax for trial in trials])
    width_min = max([trial.xmax for trial in trials])
    max_max = max([trial.max for trial in trials])
    min_min = min([trial.min for trial in trials])
    heights = [trial.xmax for trial in trials]
    heights[-1] = heights[-1]*1.5
    heights[0] = heights[0]*2
    fig, axs = plt.subplots(6, 1,
                            sharey='col',
                            # layout="compressed",
                            figsize=(24.0, 9.0),
                            height_ratios=heights)
    for trial in trials:
        rst=axs[idx].imshow(trial.history_data, vmin=min_min, vmax=max_max, cmap="RdBu")
        axs[idx].set_aspect('equal', anchor='W')
        # axs[idx].set_aspect('equal', adjustable='box', anchor='W')
        # axs[idx].set_xlim(0, trial.xmax)
        axs[idx].set_ylabel("{} m/s".format(trial.vel))
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        nm_to_dx = lambda x: x/1.25/0.6
        scale_in_mu = 0.5
        ob = AnchoredHScaleBar(
            ax=axs[idx],
            size=nm_to_dx(scale_in_mu*1000),
            label="", # "{} $\mu$m".format(scale_in_mu),
            loc=4,
            frameon=False,
            pad=0.6,
            sep=2,
            linekw=dict(color="black"),
        )
        axs[idx].add_artist(ob)
        if idx == len(trials)-1:
            plt.colorbar(rst, location='bottom')
        idx = idx + 1
    fig.savefig(plots_dir + "composite.png")
    return None

def make_visualizations(trial):
    make_history(trial)
    make_evolution(trial)
    make_crosssection(trial)
    return None

def make_history(trial):
    print("making history for", trial.tag)
    # plot history data using imshow
    hist_fig, hist_ax = plt.subplots()
    hist_artist = hist_ax.imshow(
        trial.history_data,
        vmin=trial.min-0.02,
        vmax=trial.max+0.02,
        cmap="RdBu"
    )
    # hist_ax.set_title("full history of alloy solidification")
    hist_fig.set_figwidth(16)
    hist_fig.set_figheight(3)
    # hist_ax.set_yticks(np.arange(0, ymax, 1000))
    # hist_ax.set_xticks(np.arange(0, xmax, 1000))
    hist_ax.set_aspect("equal")
    hist_ax.set_xlim(0, trial.xmax)
    hist_ax.set_xticks([])
    hist_ax.set_ylim(0, trial.ymax)
    hist_ax.set_yticks([])
    for axis in ["top", "bottom", "left", "right"]:
        hist_ax.spines[axis].set_linewidth(0)
    hist_ax.set_title(f"$V = {trial.vel}$ m/s $G = {trial.grad}$ K/nm")
    plt.colorbar(hist_artist, location='bottom', shrink=0.25)
    nm_to_dx = lambda x: x/1.25/0.6
    scale_in_mu = 0.5
    ob = AnchoredHScaleBar(
        size=nm_to_dx(scale_in_mu*1000),
        label="{} $\mu$m".format(scale_in_mu),
        loc=4,
        frameon=False,
        pad=0.6,
        sep=2,
        linekw=dict(color="black"),
    )
    hist_ax.add_artist(ob)
    # hist_ax.axvline(trial.xmax-1000)
    # save snapshot animation
    hist_fig.savefig(plots_dir + trial.tag + "history.png")
    return None

def make_evolution(trial):
    print("making evolution for", trial.tag)
    # load snapshot data
    # plot single snapshot using imshow
    snap_fig, snap_ax = plt.subplots(1,2)
    snap_artist = snap_ax[0].imshow(trial.init_data, vmin=trial.min-0.02, vmax=trial.max+0.02, cmap="RdBu")
    profile_artist = snap_ax[1].plot(trial.init_data[1245])[0]
    snap_ax[1].set_ylim(trial.min, trial.max)
    # write step function for a video of the snapshots
    def func(frame):
        dat = thicken(np.loadtxt(data_dir + trial.tag + "c_{}.txt".format(frame)))
        d_min = np.min(dat)
        d_max = np.max(dat)
        snap_artist.set_data(dat)
        snap_artist.set_clim(vmin=d_min-0.02, vmax=d_max+0.02)
        profile_artist.set_ydata(dat[1245,:])
        snap_ax[0].set_title("frame {}".format(frame))
        return [snap_artist]
    # add colorbars
    # declare funcAnimation
    anim = ani.FuncAnimation(snap_fig, func, trial.frames, interval=frame_delay)
    # save snapshot animation
    anim.save(plots_dir + trial.tag + "evolution.gif")
    return None

def make_crosssection(trial):
    print("making cross section for", trial.tag)
    steps_back = 1000
    start = 0
    end = 0
    prev = 0
    first = True
    track = False
    cooldown = True
    # this is a nasty hack to try and record the location of two peaks in the data
    for idx, conc in enumerate(trial.history_data[:,trial.xmax-steps_back]):
        if conc > 0.65:
            track = True
        else:
            cooldown = True
        if conc < prev:
            track = False
            cooldown = False
            if start != 0:
                first = False
            if end != 0:
                break

        if track and cooldown :
            if first:
                start = idx
            else:
                end = idx
        prev = conc
    print(start, end)
    fig, ax = plt.subplots()
    ys = np.arange(0, trial.ymax, 1)
    ax.plot(ys, trial.history_data[:, trial.xmax-steps_back])
    ax.axvline(start)
    ax.axvline(end)
    ax.set_title("Width = {} nm".format((end-start)*1.25*0.6))
    fig.savefig(plots_dir + trial.tag + "cross_section.png")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

def ticks_adjust(ax, axis, ticks_val, adjust_factor):
    if axis == 'x':
        ax.set_xticks(ticks_val)
        ax.set_xticklabels([int(v/adjust_factor) for v in ticks_val])
    elif axis == 'y':
        ax.set_yticks(ticks_val)
        ax.set_yticklabels([int(v/adjust_factor) for v in ticks_val])

def plt_set_size(SMALL_SIZE):
    '''
    set the size of the plt values
    '''
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)   # fontsize of the figure title

def mark_inset_zoom(ax, ax_inset, inset_bounds, **kwargs):
    inset_coor = get_coorlim(ax_inset)
    [x0, y0, w, h] = inset_coor
    [x1, y1, w1, h1] = inset_bounds
    plot_box(ax, inset_coor, **kwargs)

    #only for side to side
    ax.plot([x0+w, x1], [y0+h, y1+h1], **kwargs)
    ax.plot([x0+w, x1], [y0, y1], **kwargs)

def get_coorlim(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = x1-x0
    h = y1-y0

    return [x0, y0, w, h]

def plot_box(ax, loc, **kwargs):
    x0, y0, w, h = loc
    ax.plot([x0, x0], [y0+h, y0], **kwargs)
    ax.plot([x0, x0+w], [y0, y0], **kwargs)
    ax.plot([x0+w, x0+w], [y0, y0+h], **kwargs)
    ax.plot([x0+w, x0], [y0+h, y0+h], **kwargs)    

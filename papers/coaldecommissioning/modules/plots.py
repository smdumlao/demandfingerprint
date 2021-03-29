'''
Collection of functions for the plotting in this project
'''
# plotting-related libraries
import calendar
import temphelper
from matplotlib.lines import Line2D


cols2 = ['warmpeak', 'warmmean', 'coldmean', 'coldpeak']

def legend_elements_clustering():
    legend_elements = []

    markers = ['*', '+', 'D', 'x', 'o', '1', 's', '2', 'v', '3', '^', '4']
    #lcolor = ["darkviolet","blue","forestgreen", "darkorange", "crimson"]
    linestyles = ['']*12
    labels = [calendar.month_abbr[i] for i in range (1,13)]
    for ls, label, marker in zip(linestyles, labels, markers):
        legend_elements.append(Line2D([0], [0], color='k', lw=4, linestyle=ls,
                                label=label, marker=marker))
    return legend_elements

def legend_elements_year_rep():
    legend_elements = []
    subtitle = ['MM', 'MS', 'SM', 'SS', 'E-', '-E']

    years = [2014,1991,2016,2018,2013,2012]
    lcolor6 = ["darkorange", "forestgreen","gold",
                "saddlebrown", "red", "royalblue"]
    linestyles_a = ['-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]
    linestyles =['-']*6+ linestyles_a
    linewidth = [6]*6 #+ [2]*4
    labels = [f'{y} {s}' for y, s in zip(years, subtitle)]

    for c, ls, label, lw in zip(lcolor6, linestyles, labels, linewidth):
        legend_elements.append(Line2D([0], [0], color=c, lw=lw,
                                linestyle=ls, label=label))
    return legend_elements

def legend_elements_rep_year():
    '''
    s1
    '''
    legend_elements = []
    subtitle = ['MM', 'MS', 'SM', 'SS', 'E-', '-E']

    years = [2014,1991,2016,2018,2013,2012]
    lcolor6 = ["darkorange", "forestgreen","gold", "saddlebrown", "red", "royalblue"] + ['k']*4
    cols2 = ['warmpeak', 'warmmean', 'coldmean', 'coldpeak']
    linestyles_a = ['-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]
    linestyles =['-']*6+ linestyles_a
    linewidth = [6]*6 + [2]*4
    labels = [f'{str(y)[2:]}{s}' for y, s in zip(years, subtitle)] + ['warm peak', 'warm mean', 'cold mean', 'cold peak']

    for c, ls, label, lw in zip(lcolor6, linestyles, labels, linewidth):
        legend_elements.append(Line2D([0], [0], color=c, lw=lw, linestyle=ls, label=label))

    return legend_elements



def legend_elements_rep(rep, years):
    '''
    s1
    '''
    legend_elements = []
    cluster_color = ['darkorange', 'forestgreen', 'gold', 'saddlebrown', 'red', 'royalblue',
                         'orchid', 'darkviolet']
    linestyles_a = ['-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]
    linestyles =['-']*(len(years)+1) + linestyles_a
    linewidth = [6]*(len(years)+1) + [2]*4
    labels = [rep] + years + ['warm peak', 'warm mean', 'cold mean', 'cold peak']
    colors = ['k'] + cluster_color[:len(years)] + ['k']*4
    for c, ls, label, lw in zip(colors, linestyles, labels, linewidth):
        legend_elements.append(Line2D([0], [0], color=c, lw=lw, linestyle=ls, label=label))

    return legend_elements

def plot_year_cluster(ax, kyushu_temp, year_rep, year0):
    '''
    s1
    '''

    year_cluster = year_rep[year_rep['rep']==year0].index.to_list()
    cluster_color = ['darkorange', 'forestgreen', 'gold', 'saddlebrown', 'red', 'royalblue',
                     'orchid', 'darkviolet']
    linestyles_a = ['-', '-.', '--', (0, (3, 1, 1, 1, 1, 1))]
    year_cluster.remove(year0)

    for year, c in zip(year_cluster, cluster_color):
        dfx = temphelper.get_temp_offset_stat(kyushu_temp, year)
        for col, ls in zip(cols2, linestyles_a):
            dfx[col].plot(ax=ax, color=c, linestyle=ls, legend=False, lw=2)

    dfx = temphelper.get_temp_offset_stat(kyushu_temp, year0)
    for col, ls in zip(cols2, linestyles_a):
        dfx[col].plot(ax=ax, color='k', linestyle=ls, legend=False, lw=2)

    ax.legend(handles = legend_elements_rep(year0, year_cluster), ncol=4, loc='upper center',
              edgecolor='white', fontsize=11, framealpha = 1, columnspacing = .5, labelspacing = 0.1,
              bbox_to_anchor=[.5, 1.01]);

    ax.set_xlim(1,12)
    ax.set_ylim(-25,25)
    ax.set(xlabel = '', ylabel = '')
    ax.set_xticks(range(1,13));
    ax.set_yticks(range(-25, 26, 5))
    ax.grid()

def weather_labels():
    '''
    s1
    '''
    s_mild = '$Summer_{mild}$'
    s_severe = '$Summer_{severe}$'
    s_extreme = '$Summer_{extreme}$'

    w_mild = '$Winter_{mild}$'
    w_severe = '$Winter_{severe}$'
    w_extreme = '$Winter_{extreme}$'

    labels = [f'{s_mild} {w_mild}', f'{s_mild} {w_severe}', f'{s_severe} {w_mild}',
             f'{s_severe} {w_severe}', f'{s_extreme}', f'{w_extreme}']
    return labels

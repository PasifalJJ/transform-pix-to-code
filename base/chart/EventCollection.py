import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

# split the data into two parts
xdata1 = np.asarray([i + 1 for i in range(50)])
# xdata2 = np.asarray([i + 1 for i in range(50)])

# sort the data so it makes clean curves
xdata1.sort()
# xdata2.sort()

# create some y data points
ydata1 = np.asarray(
    [0.27941614898795986, 0.0960409951605359, 0.07290849732805271, 0.05905545257716155, 0.05196905187426173, 0.04471879503594674, 0.03965129846592947, 0.03835246860030658, 0.032829118656868835, 0.030563782984200788, 0.028492523961681494, 0.027525839380615827, 0.02507680783010259, 0.02329406224723373, 0.023138573927310658, 0.02156377889367999, 0.022904480414046927, 0.01988926243645196, 0.02022607163621151, 0.016402385112050236, 0.015140447967058542, 0.015812699095707157, 0.013368541528755913, 0.012549215616012107, 0.010658916355851962, 0.00988094416941155, 0.007379339460036432, 0.005890346594316391, 0.006009773760606364, 0.005239294687219496, 0.004394579476413165, 0.004107368853042966, 0.003783462757282719, 0.0030247354586561193, 0.0027760345602826195, 0.0026112314370075923, 0.003331581687635495, 0.003105896113592447, 0.0027444347621616433, 0.002423787940108236, 0.0022785687116473646, 0.0021472708874486616, 0.0021439722168962567, 0.0022470815670593376, 0.002452378078780793, 0.001743989281192403, 0.001744643886031925, 0.001761034206333998, 0.0018176226396165902, 0.0016283302862946732]
    # [6
    #     , 5.8
    #     , 5.6
    #     , 5.5
    #     , 5.29
    #     , 5.2
    #     , 5.1
    #     , 5
    #     , 4.9
    #     , 4.7
    #     , 4.6
    #     , 4.5
    #     , 4.4
    #     , 4.3
    #     , 4.2
    #     , 4.1
    #     , 4
    #     , 3.9
    #     , 3.8
    #     , 3.6
    #     , 3.4
    #     , 3.15
    #     , 3
    #     , 2.5
    #     , 2
    #     , 1.5
    #     , 1.1
    #     , 0.8
    #     , 0.6
    #     , 0.5
    #     , 0.48
    #     , 0.46
    #     , 0.42
    #     , 0.36
    #     , 0.32
    #     , 0.28
    #     , 0.27
    #     , 0.26
    #     , 0.25
    #     , 0.24
    #     , 0.23
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22
    #     , 0.22]
)
# ydata2 = np.asarray(
#     [6.1
#         , 5.82
#         , 5.65
#         , 5.55
#         , 5.4
#         , 5.3
#         , 5.2
#         , 5.1
#         , 4.99
#         , 4.73
#         , 4.65
#         , 4.56
#         , 4.45
#         , 4.35
#         , 4.25
#         , 4.18
#         , 4.1
#         , 3.95
#         , 3.85
#         , 3.65
#         , 3.45
#         , 3.2
#         , 3.1
#         , 3
#         , 2.8
#         , 2
#         , 1.5
#         , 1.2
#         , 1
#         , 0.8
#         , 0.65
#         , 0.63
#         , 0.612
#         , 0.6
#         , 0.58
#         , 0.56
#         , 0.54
#         , 0.53
#         , 0.51
#         , 0.48
#         , 0.46
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#         , 0.42
#
#      ])

# plot the data1
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.plot(xdata1, ydata1, color='tab:blue', label='train')
# ax.plot(xdata2, ydata2, color='tab:orange', label='val')

# create the events marking the x data points
# xevents1 = EventCollection(xdata1, color='tab:blue', linelength=0.05)
# xevents2 = EventCollection(xdata2, color='tab:orange', linelength=0.05)

# create the events marking the y data points
yevents1 = EventCollection(ydata1, color='tab:blue', linelength=0.05,
                           orientation='vertical')
# yevents2 = EventCollection(ydata2, color='tab:orange', linelength=0.05,
#                            orientation='vertical')
plt.legend(loc=0,ncol=2)
# add the events to the axis
# ax.add_collection(xevents1)
# ax.add_collection(xevents2)
ax.add_collection(yevents1)
# ax.add_collection(yevents2)

# set the limits
ax.set_xlim([1, 50])
ax.set_ylim([0.0, 0.3])

ax.set_title('Epoch50-Loss')

# display the plot
plt.show()

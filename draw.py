import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

pA = np.array([7.75, 4.25, 1.75])
pB = np.array([12.25, 7.0, 0.25])
halfSubIncs = 0.5 * np.array([1.5, 1, 0.5])
muDet = 10

sensDetSubGeom = np.array(
    [
        pB[0] - halfSubIncs[0],
        pB[0] + halfSubIncs[0],
        pB[1] - halfSubIncs[1],
        pB[1] + halfSubIncs[1],
        pB[2] - halfSubIncs[2],
        pB[2] + halfSubIncs[2],
        -1,
        muDet,
    ]
)
fig,ax=plt.subplots(1,2,figsize=(20,6))
target_rect=plt.Rectangle(pB[0:2]-halfSubIncs[0:2],2*halfSubIncs[0],2*halfSubIncs[1],color='r')
ax[0].add_patch(target_rect)
ax[0].set_aspect('equal')
ax[0].scatter([pA[0],pB[0]],[pA[1],pB[1]])
ax[0].plot([pA[0],pB[0]],[pA[1],pB[1]])

target_rect=plt.Rectangle(pB[0::2]-halfSubIncs[0::2],2*halfSubIncs[0],2*halfSubIncs[2],color='b')
ax[1].add_patch(target_rect)
ax[1].set_aspect('equal')
ax[1].scatter([pA[0],pB[0]],[pA[2],pB[2]])
ax[1].plot([pA[0],pB[0]],[pA[2],pB[2]])
fig.savefig("plot.png")
# plt.show()
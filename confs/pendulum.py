import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib.animation import Animation, FuncAnimation

import IPython

# definition of the matrices and example of control
g = 9.81
dt = 0.01

def animate_robot(x0, u):
    """
    This function makes an animation showing the behavior of the pendulum
    takes as input the result of a simulation - dt is the sampling time (0.1s normally)
    """

    assert(u.shape[0]==1)
    assert(x0.shape[0]==2)
    N = u.shape[1] + 1
    x = np.zeros((2,N))
    x[:,0] = x0[:,0]
    for i in range(N-1):
        x[0,i+1] = x[0,i] + dt * x[1,i]
        x[1,i+1] = x[1,i] + dt * (u[0,i] - g * np.sin(x[0,i]))
    
    # here we check if we need to down-sample the data for display
    #downsampling (we want 100ms DT or higher)
    min_dt = 0.1
    if(dt < min_dt):
        steps = int(min_dt/dt)
        use_dt = int(min_dt * 1000)
    else:
        steps = 1
        use_dt = int(dt * 1000)
    plotx = x[:,::steps]

    fig, ax = plt.subplots(figsize=[6,6])
    ax.set_xlim([-1.3,1.3])
    ax.set_ylim([-1.3,1.3])
    ax.set_autoscale_on(False)
    ax.grid()

    list_of_lines = []

    #create the pendulum
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'o', lw=2)
    list_of_lines.append(line)

    def animate(i):
        for l in list_of_lines: #reset all lines
            l.set_data([],[])

        x_pend = np.sin(plotx[0,i])
        y_pend = -np.cos(plotx[0,i])

        list_of_lines[0].set_data([0., x_pend], [0., y_pend])
        list_of_lines[1].set_data([x_pend, x_pend], [y_pend, y_pend])

        return list_of_lines

    def init():
        return animate(0)

    ani = FuncAnimation(fig, animate, np.arange(0, len(plotx[0,:])),
        interval=use_dt, blit=True, init_func=init)

    # Show the animation
    plt.show()

    # This following is used for Jupyter notebooks to display the animation
    #plt.close(fig)
    #plt.close(ani._fig)
    #IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))

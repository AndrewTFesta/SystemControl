"""
@title
@description
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


# Sometimes, the animation will fail to update when run from an IDE.
# In PyCharm, a possible fix is to disable scientific plotting.
# File -> Settings -> Tools -> Python Scientific
#   Make sure the box "Show plots in tool window" is unchecked.
# If the animation still fails to update, try again with the following
# line uncommented.
# matplotlib.use("TkAgg")

# def FFT(x,y):
#     X = (x[-1]-x[0])/len(y)
#     f = np.linspace(-2*np.pi/X/2,2*np.pi/X/2,len(y))
#     F = np.fft.fftshift(np.fft.fft(y))/np.sqrt(len(y))
#     return(f,F)

class MatplotlibVisualizer(animation.FuncAnimation):

    def __init__(self, data_source: DataSource, subject: str):
        self.data_source = data_source
        self.subject = subject
        self.num_seconds = 5

        self._data_iterator = self.data_source.__iter__()
        self.update_delay = 1. / self.data_source.sample_freq
        self.num_samples = self.data_source.sample_freq * self.num_seconds
        self.channel_names = self.data_source.coi
        self.event_names = self.data_source.event_names

        self.max_abs_val = 20E-5

        fig = plt.figure(figsize=(8, 8), facecolor='black')
        '''
            *pos* is a three digit integer, where the first digit is the
            number of rows, the second the number of columns, and the third
            the index of the subplot. i.e. fig.add_subplot(235) is the same as
            fig.add_subplot(2, 3, 5). Note that all integers must be less than
            10 for this form to work.
        '''
        # Add a subplot with no frame
        ax = plt.subplot(111, frameon=False)

        # Generate random data
        self.data = np.random.uniform(0, 1, (64, 75))
        X = np.linspace(-1, 1, self.data.shape[-1])
        self.G = 1.5 * np.exp(-4 * X * X)

        # Generate line plots
        self.lines = []
        for i in range(len(self.data)):
            # Small reduction of the X extents to get a cheap perspective effect
            xscale = 1 - i / 200.
            # Same for linewidth (thicker strokes on bottom)
            lw = 1.5 - i / 100.0
            line, = ax.plot(xscale * X, i + self.G * self.data[i], color="w", lw=lw)
            self.lines.append(line)

        # Set y limit (or first line is cropped because of thickness)
        ax.set_ylim(-1, 70)

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # 2 part titles to get different font weights
        ax.text(0.5, 1.0, "MATPLOTLIB ", transform=ax.transAxes,
                ha="right", va="bottom", color="w",
                family="sans-serif", fontweight="light", fontsize=16)
        ax.text(0.5, 1.0, "UNCHAINED", transform=ax.transAxes,
                ha="left", va="bottom", color="w",
                family="sans-serif", fontweight="bold", fontsize=16)

        animation.FuncAnimation.__init__(self, fig, func=self.update, interval=50)
        return

    def update(self, count):
        # Shift all data to the right
        self.data[:, 1:] = self.data[:, :-1]

        # Fill-in new values
        self.data[:, 0] = np.random.uniform(0, 1, len(self.data))

        # Update data
        for i in range(len(self.data)):
            self.lines[i].set_ydata(i + self.G * self.data[i])

        # Return modified artists
        return self.lines




def main():
    # physio_ds = PhysioDataSource()
    #
    # vid_writer = animation.writers['ffmpeg']
    # writer = vid_writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    #
    # save_dir = os.path.join(DATA_DIR, 'output', 'animations')
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # save_name = os.path.join(save_dir, f'test_file.mp4')
    #
    # eeg_visualizer = MatplotlibVisualizer(physio_ds, physio_ds.ascended_being)
    # eeg_visualizer.save('save_name.mp4', writer=writer)
    # plt.show()

    # Settings
    video_file = "myvid.mp4"
    clear_frames = False  # Should it clear the figure between each frame?
    fps = 15

    # Output video writer
    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = ffmpeg_writer(fps=fps, metadata=metadata)

    fig = plt.figure()
    x = np.arange(0, 10, 0.1)
    with writer.saving(fig, video_file, 100):
        for i in np.arange(1, 10, 0.1):
            y = np.sin(x + i)
            if clear_frames:
                fig.clear()
            ax, = plt.plot(x, y, 'r-', linestyle="solid", linewidth=1)
            writer.grab_frame()
    return


if __name__ == '__main__':
    main()

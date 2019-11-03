"""
@title
@description
"""

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.layouts import gridplot
from bokeh.models import FuncTickFormatter
from bokeh.palettes import Dark2
from bokeh.plotting import figure
from bokeh.server.server import Server

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


class BokehVisualizer:
    MAX_WIDTH = 800
    MAX_HEIGHT = 800
    POINTS_PRE_UPDATE = 1

    def __init__(self, data_source: DataSource, subject: str):
        self.data_source = data_source
        self.subject = subject

        self._data_iterator = self.data_source.__iter__()
        self.update_delay = 1. / self.data_source.sample_freq

        self.x_len = int(self.data_source.sample_freq)
        self.channel_names = self.data_source.coi
        self.event_names = self.data_source.event_names

        self.max_abs_val = 20E-5

        self.signal_stream_figure = None
        self.event_stream_figure = None

        self.lines = {}
        self.line_ds = {}

        self.ray = None
        self.ray_ds = None

        self.apps = {'/': Application(FunctionHandler(self.make_doc))}
        self.server = Server(self.apps, port=5001)
        return

    def make_doc(self, doc):
        self.build_signal_stream_figure()
        self.build_event_stream_figure()

        doc.add_periodic_callback(self.update, self.update_delay)
        doc.title = 'EEG Stream'

        grid = gridplot(
            [
                [self.signal_stream_figure, None],
                # [None, self.event_stream_figure]
            ],
            plot_width=self.MAX_WIDTH * 2, plot_height=self.MAX_HEIGHT * 2
        )
        doc.add_root(grid)
        return

    def build_event_stream_figure(self):
        title_str = f'{self.data_source.name} data' # todo fix
        sep = ': '
        for each_ch in self.channel_names:
            title_str += f'{sep}{each_ch}'
            sep = ', '

        self.event_stream_figure = figure(
            plot_width=self.MAX_WIDTH,
            plot_height=self.MAX_HEIGHT,
            title=title_str,
            sizing_mode='scale_height',
            # output_backend='webgl'
        )
        self.event_stream_figure.x_range.follow = "end"
        self.event_stream_figure.x_range.follow_interval = self.x_len
        self.event_stream_figure.x_range.range_padding = 0

        # implicitly assumes a parameter tick
        #       tick: x value of this point on the plot
        # requires installation of package: pscript
        x_tick_code = f"""
            function format_x_tick(x) {{
                return tick / {self.data_source.sample_freq:0.2f}
            }}
            return format_x_tick(tick)
        """
        self.event_stream_figure.xaxis.formatter = FuncTickFormatter(code=x_tick_code)

        return

    def build_signal_stream_figure(self):
        title_str = f'{self.data_source.name} data'
        sep = ': '
        for each_ch in self.channel_names:
            title_str += f'{sep}{each_ch}'
            sep = ', '

        self.signal_stream_figure = figure(
            plot_width=self.MAX_WIDTH,
            plot_height=self.MAX_HEIGHT,
            title=title_str,
            sizing_mode='scale_height',
            # output_backend='webgl'
        )
        self.signal_stream_figure.x_range.follow = "end"
        self.signal_stream_figure.x_range.follow_interval = self.x_len
        self.signal_stream_figure.x_range.range_padding = 0

        # implicitly assumes a parameter tick
        #       tick: x value of this point on the plot
        # requires installation of package: pscript
        x_tick_code = f"""
            function format_x_tick(x) {{
                return tick / {self.data_source.sample_freq:0.2f}
            }}
            return format_x_tick(tick)
        """
        self.signal_stream_figure.xaxis.formatter = FuncTickFormatter(code=x_tick_code)

        for each_channel in self.channel_names:
            # new_line = self.figure.segment(
            #     x0=[], y0=[], x1=[], y1=[], line_color=[],
            #     line_width=2
            # )
            new_line = self.signal_stream_figure.line(x=[], y=[], line_color='red', line_width=2)
            self.lines[each_channel] = {'line': new_line, 'ds': new_line.data_source}
            self.line_ds[each_channel] = new_line.data_source

        # self.ray = self.signal_stream_figure.ray(
        #     x=[], y=[], angle=[],
        #     length=self.MAX_HEIGHT * 2,
        #     angle_units="deg",
        #     color="#FB8072",
        #     line_width=2
        # )
        # self.ray_ds = self.ray.data_source
        return
    # def get_event_from_index(self, idx: int) -> PhysioDataSource.event_names:
    #     time_val = self.data_source.idx_to_time(self.raw_data, idx)
    #     onset_list = self.events.onset
    #     desc_list = self.events.description
    #     for onset_idx, each_val in enumerate(onset_list):
    #         if each_val >= time_val:
    #             event_str = desc_list[onset_idx]
    #             enum_val = PhysioEvent[event_str]
    #             break
    #     else:
    #         enum_val = PhysioDataSource.event_names[0]
    #     return enum_val

    # noinspection PyTypeChecker
    def update(self):
        # curr_event = self.get_event_from_index(self.index)
        curr_event: str = self.event_names[0]
        num_events: int = len(self.event_names)

        colors = Dark2[num_events]
        eve_color = colors[0]

        next_sample = next(self._data_iterator)
        sample_idx = next_sample['idx']
        sample_data = next_sample['data']

        for ch_idx, ((ch_name, data_source), ch_val) in enumerate(zip(self.line_ds.items(), sample_data)):
            vertical_offset = self.max_abs_val * (ch_idx - 1)
            new_entry = {
                'x': [sample_idx],
                'y': [ch_val + vertical_offset]
            }
            data_source.stream(new_entry)

        # is_event_onset = any(each_time[0] == self.index for each_time in self.event_times)
        # if is_event_onset:
        #     # draw 2 new rays, starting at origin and going in opposite directions
        #     # appears that a single line is drawn over entire area
        #     new_entry = {
        #         'x': [self.index, self.index],
        #         'y': [0, 0],
        #         'angle': [90, 270]
        #     }
        #     self.ray_ds.stream(new_entry)
        # self.index += 1
        return

    def run(self, open_tab: bool = False):
        if open_tab:
            self.server.show('/')
        self.server.run_until_shutdown()
        return


def main():
    physio_ds = PhysioDataSource()

    eeg_visualizer = BokehVisualizer(physio_ds, physio_ds.ascended_being)
    eeg_visualizer.run(open_tab=True)
    return


if __name__ == '__main__':
    main()

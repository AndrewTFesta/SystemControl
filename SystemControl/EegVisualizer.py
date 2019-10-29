"""
@title
@description
"""

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models import FuncTickFormatter
from bokeh.palettes import Dark2
from bokeh.plotting import figure
from bokeh.server.server import Server
from mne import events_from_annotations

from SystemControl import DATABASE_URL
from SystemControl.DataSource import SqlDb, DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource, PhysioEvent


class EegVisualizer:
    MAX_WIDTH = 800
    MAX_HEIGHT = 800
    POINTS_PRE_UPDATE = 1

    def __init__(self, data_source: DataSource, subject: int = 1):
        self.data_source = data_source
        self.subject = subject

        self.raw_data = self.data_source.get_mi_right_left(self.subject)
        self.data = self.data_source.get_data(self.raw_data)
        self.events = self.data_source.get_annotations(self.raw_data)

        evts = events_from_annotations(self.raw_data)
        self.event_times = evts[0]

        self.sfreq = self.raw_data.info['sfreq']
        self.update_delay = 1. / self.sfreq

        self.x_len = int(self.sfreq)
        self.channel_names = self.data_source.get_channel_names(self.raw_data)

        self.max_abs_val = 20E-5

        self.figure = None
        self.index = 0

        self.lines = {}
        self.line_ds = {}

        self.ray = None
        self.ray_ds = None

        self.apps = {'/': Application(FunctionHandler(self.make_doc))}
        self.server = Server(self.apps, port=5001)
        return

    def make_doc(self, doc):
        title_str = f'{self.data_source} data'
        sep = ': '
        for each_ch in self.channel_names:
            title_str += f'{sep}{each_ch}'
            sep = ', '
        self.figure = figure(
            plot_width=self.MAX_WIDTH,
            plot_height=self.MAX_HEIGHT,
            title=title_str,
            sizing_mode='scale_height'
        )
        self.figure.x_range.follow = "end"
        self.figure.x_range.follow_interval = self.x_len
        self.figure.x_range.range_padding = 0

        # implicitly assumes a parameter tick
        #       tick: x value of this point on the plot
        # requires installation of package: pscript
        x_tick_code = """
            return tick / {:0.2f}
        """.format(self.sfreq)
        self.figure.xaxis.formatter = FuncTickFormatter(code=x_tick_code)

        # todo set yticks based on vertical offset
        # y_tick_code = f"""
        #     function format_y_tick(x) {{
        #         return (tick % {self.max_abs_val}).toExponential(2);
        #     }}
        #     return format_y_tick(tick)
        # """
        # self.figure.yaxis.formatter = FuncTickFormatter(code=y_tick_code)

        for each_channel in self.channel_names:
            new_line = self.figure.segment(
                x0=[], y0=[], x1=[], y1=[], line_color=[],
                line_width=2
            )
            self.lines[each_channel] = {'line': new_line, 'ds': new_line.data_source}
            self.line_ds[each_channel] = new_line.data_source

        self.ray = self.figure.ray(
            x=[], y=[], angle=[],
            length=self.MAX_HEIGHT * 2,
            angle_units="deg",
            color="#FB8072",
            line_width=2
        )
        self.ray_ds = self.ray.data_source

        self.index = 0
        doc.add_periodic_callback(self.update, self.update_delay)
        doc.title = 'EEG Stream'
        doc.add_root(self.figure)
        return

    def get_event_from_index(self, idx: int) -> PhysioEvent:
        time_val = self.data_source.idx_to_time(self.raw_data, idx)
        onset_list = self.events.onset
        desc_list = self.events.description
        for onset_idx, each_val in enumerate(onset_list):
            if each_val >= time_val:
                event_str = desc_list[onset_idx]
                enum_val = PhysioEvent[event_str]
                break
        else:
            enum_val = PhysioEvent.T0
        return enum_val

    # noinspection PyTypeChecker
    def update(self):
        curr_event = self.get_event_from_index(self.index)
        num_events = len(PhysioEvent)

        colors = Dark2[num_events]
        eve_color = colors[curr_event.value]

        for loop_idx, entry in enumerate(zip(self.line_ds.items(), self.data)):
            d_source = entry[0][1]
            data_entry = entry[1]

            vertical_offset = self.max_abs_val * (loop_idx - 1)
            prev_xs = list(range(self.index, self.index + self.POINTS_PRE_UPDATE))
            next_xs = list(range(self.index + 1, self.index + 1 + self.POINTS_PRE_UPDATE))
            prev_ys = data_entry[self.index:self.index + self.POINTS_PRE_UPDATE] + vertical_offset
            next_ys = data_entry[self.index + 1:self.index + 1 + self.POINTS_PRE_UPDATE] + vertical_offset

            new_entry = {
                'x0': prev_xs,
                'x1': next_xs,
                'y0': prev_ys,
                'y1': next_ys,
                'line_color': [eve_color] * self.POINTS_PRE_UPDATE
            }
            d_source.stream(new_entry, self.x_len * 2)

        is_event_onset = any(each_time[0] == self.index for each_time in self.event_times)
        if is_event_onset:
            # draw 2 new rays, starting at origin and going in opposite directions
            # appears that a single line is drawn over entire area
            new_entry = {
                'x': [self.index, self.index],
                'y': [0, 0],
                'angle': [90, 270]
            }
            self.ray_ds.stream(new_entry)
        self.index += 1
        return

    def run(self, open_tab: bool = False):
        if open_tab:
            self.server.show('/')
        self.server.run_until_shutdown()
        return

    def plot_data(self, tmin, tmax, with_events: bool = False):
        # todo
        return


def main():
    subject = 1

    db_path = DATABASE_URL
    database = SqlDb.SqlDb(db_path)
    physio_ds = PhysioDataSource(database)

    eeg_visualizer = EegVisualizer(physio_ds, subject)
    eeg_visualizer.run()
    return


if __name__ == '__main__':
    main()

"""Echarts python wrapper."""

from __future__ import division

import os
import uuid
import webbrowser
from tempfile import NamedTemporaryFile
from collections import defaultdict
import json
from datetime import datetime, date, time


# pylint: disable=redefined-builtin
from builtins import zip

import numpy as np
import pandas as pd

from jinja2 import Environment, FileSystemLoader


__version__ = '0.0.1-dev'


ECHARTS_VERSION = '3.5.3'


def _recursive_dict(*args):
    recursive_factory = lambda: defaultdict(recursive_factory)
    return defaultdict(recursive_factory, *args)


def _labels(base='trace'):
    i = 0
    while True:
        yield base + ' ' + str(i)
        i += 1


def _detect_notebook():
    """
    This isn't 100% correct but seems good enough

    Returns
    -------
    bool
        True if it detects this is a notebook, otherwise False.
    """
    try:
        from IPython import get_ipython
        from ipykernel import zmqshell
    except ImportError:
        return False
    kernel = get_ipython()
    return isinstance(kernel, zmqshell.ZMQInteractiveShell)


def _merge_layout(x, y):
    z = y.copy()
    if 'shapes' in z and 'shapes' in x:
        x['shapes'] += z['shapes']
    z.update(x)
    return z


def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def is_datetime(x):
    return isinstance(x, (datetime, date, time))


def check_type(x):
    """Simple heuristic."""
    if is_number(x):
        return 'value'
    elif is_datetime(x):
        return 'time'
    return 'category'


def _try_pydatetime(x):
    """Opportunistically try to convert to pandas time indexes
    since plotly doesn't know how to handle them.
    """
    try:
        x = x.to_pydatetime()
    except AttributeError:
        pass
    return x


def json_conversion(obj):
    """Encode additional objects to JSON."""
    try:
        # numpy isn't an explicit dependency of bowtie
        # so we can't assume it's available
        import numpy as np
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
    except ImportError:
        pass

    try:
        # pandas isn't an explicit dependency of bowtie
        # so we can't assume it's available
        import pandas as pd
        if isinstance(obj, pd.Index):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, (datetime, time, date)):
        return obj.isoformat()
    raise TypeError('Not sure how to serialize {} of type {}'.format(obj, type(obj)))


def jdumps(data):
    """Encode Python object to JSON with additional encoders."""
    return json.dumps(data, default=json_conversion)


class Chart(object):
    """
    Echart base class, usually this object will get created
    by from a function.
    """

    def __init__(self, data=None): # data=None, layout=None, repr_plot=True):
        if data is None:
            data = {}
        self.chart = _recursive_dict(data)
        # self.repr_plot = repr_plot

        # sensible defaults
        self.chart['toolbox']['feature']['dataZoom']['show'] = True
        self.chart['toolbox']['feature']['dataZoom']['title'] = 'Zoom'
        self.chart['toolbox']['feature']['saveAsImage']['show'] = True
        self.chart['toolbox']['feature']['saveAsImage']['title'] = 'Save'
        self.chart['tooltip']['show'] = True
        self.tooltip()

    def __add__(self, other):
        self.chart['series'] += other.chart['series']
        # self.layout = _merge_layout(self.layout, other.layout)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def animation(self, enable):
        self.chart['animation'] = enable

    def width(self, value):
        """Sets the width of the plot in pixels.

        Parameters
        ----------
        value : int
            Width of the plot in pixels.

        Returns
        -------
        Chart

        """
        self.layout['width'] = value
        return self

    def height(self, value):
        """Sets the height of the plot in pixels.

        Parameters
        ----------
        value : int
            Height of the plot in pixels.

        Returns
        -------
        Chart

        """
        self.layout['height'] = value
        return self

    def group(self):
        """Sets bar graph display mode to "grouped".

        Returns
        -------
        Chart

        """
        self.layout['barmode'] = 'group'
        return self

    def stack(self):
        """Sets bar graph display mode to "stacked".

        Returns
        -------
        Chart

        """
        self.layout['barmode'] = 'stack'
        return self

    def xlabel(self, label):
        """Sets the x-axis title.

        Parameters
        ----------
        value : str
            Label for the x-axis

        Returns
        -------
        Chart

        """
        self.chart['xAxis'][0]['name'] = label
        return self

    def ylabel(self, label):
        """Sets the y-axis title.

        Parameters
        ----------
        value : str
            Label for the y-axis
        index : int
            Y-axis index

        Returns
        -------
        Chart

        """
        self.chart['yAxis'][0]['name'] = label
        return self

    def xtickangle(self, angle):
        """Sets the angle of the x-axis tick labels.

        Parameters
        ----------
        value : int
            Angle in degrees

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['tickangle'] = angle
        return self

    def ytickangle(self, angle, index=1):
        """Sets the angle of the y-axis tick labels.

        Parameters
        ----------
        value : int
            Angle in degrees
        index : int, optional
            Y-axis index

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickangle'] = angle
        return self

    def xlabelsize(self, size):
        """Set the size of the label

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['titlefont']['size'] = size
        return self

    def ylabelsize(self, size, index=1):
        """Set the size of the label

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['titlefont']['size'] = size
        return self

    def xticksize(self, size):
        """Set the tick font size

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['xaxis']['tickfont']['size'] = size
        return self

    def yticksize(self, size, index=1):
        """Set the tick font size

        Parameters
        ----------
        size : int

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickfont']['size'] = size
        return self

    def ytickvals(self, values, index=1):
        """Set the tick values

        Parameters
        ----------
        values : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['tickvals'] = values
        return self

    def yticktext(self, labels, index=1):
        """Set the tick labels

        Parameters
        ----------
        labels : array-like

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['ticktext'] = labels
        return self

    def xlim(self, low, high):
        """Set xaxis limits

        Parameters
        ----------
        low : number
        high : number

        Returns
        -------
        Chart

        """
        self.chart['xAxis'][0]['min'] = low
        self.chart['xAxis'][0]['max'] = high
        return self

    def ylim(self, low, high):
        """Set yaxis limits

        Parameters
        ----------
        low : number
        high : number
        index : int, optional

        Returns
        -------
        Chart

        """
        self.chart['yAxis'][0]['min'] = low
        self.chart['yAxis'][0]['max'] = high
        return self

    def xdtick(self, dtick):
        self.layout['xaxis']['dtick'] = dtick
        return self

    def ydtick(self, dtick, index=1):
        self.layout['yaxis' + str(index)]['dtick'] = dtick
        return self

    def xnticks(self, nticks):
        self.chart['xAxis'][0]['splitNumber'] = nticks
        return self

    def ynticks(self, nticks):
        self.chart['yAxis'][0]['splitNumber'] = nticks
        return self

    def tooltip(self, show=True):
        self.chart['tooltip']['show'] = show

    def tooltip_item(self):
        self.chart['tooltip']['trigger'] = 'item'

    def tooltip_axis(self):
        self.chart['tooltip']['trigger'] = 'axis'

    def legend(self, show=True):
        self.chart['legend']['show'] = show
        labels = [s['name'] for s in self.chart['series']]
        self.chart['legend']['data'] = labels

    def title(self, string):
        """Sets the title of the plot.

        Parameters
        ----------
        string : str

        Returns
        -------
        Chart

        """
        self.chart['title']['text'] = string
        return self

    def subtext(self, string):
        """Sets the subtext of the plot.

        Parameters
        ----------
        string : str

        Returns
        -------
        Chart

        """
        self.chart['title']['subtext'] = string
        return self

    def __repr__(self):
        if self.repr_plot:
            self.show(filename=None, auto_open=False)
        return super(Chart, self).__repr__()

    @property
    def _html(self):
        env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
            trim_blocks=True,
            lstrip_blocks=True
        )

        html = env.get_template('chart.j2')
        return html.render(
            version=ECHARTS_VERSION,
            options=jdumps(self.chart)
        )

    def show(self):
        # TODO notebook integration
        is_notebook = _detect_notebook()
        if not is_notebook:
            with NamedTemporaryFile(prefix='echarts', suffix='.html', mode='w+', delete=False) as f:
                f.write(self._html)
                fname = f.name
            webbrowser.open('file://' + fname)

        else:
            div = """
<div id="{chartid}" style="width:800px; height:535px;"></div>
<script>
    require.config({{
         paths:{{
            echarts: '//cdnjs.cloudflare.com/ajax/libs/echarts/3.5.3/echarts.min'
         }}
    }});
    require(['echarts'],function(ec){{
    var myChart = ec.init(document.getElementById("{chartid}"));
                var options = {options};
                myChart.setOption(options);
    }});
</script>
"""
            divid = uuid.uuid4()
            data = div.format(
                chartid=divid,
                options=jdumps(self.chart)
            )
            from IPython.display import Javascript, HTML, display
            return HTML(data)

        # is_notebook = _detect_notebook()
        # kargs = {}
        # if is_notebook:
        #     py.init_notebook_mode()
        #     plot = py.iplot
        # else:
        #     plot = py.plot
        #     if filename is None:
        #         filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        #     kargs['filename'] = filename
        #     kargs['auto_open'] = auto_open
        #
        # self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        # plot(self.figure_, show_link=show_link, **kargs)

    def save(self, filename=None, output='file', plotlyjs=True):
        # if
        with NamedTemporaryFile(prefix='echarts', suffix='.html', mode='w+', delete=False) as f:
            f.write(self._html)
            fname = f.name
        # if filename is None:
        #     filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        # self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        # NOTE: this doesn't work for output 'div'
        # py.plot(self.figure_, show_link=show_link, filename=filename, auto_open=auto_open,
        #         output_type=output, include_plotlyjs=plotlyjs)
        return filename


def _simple_chart(x=None, y=None, name=None, color=None, width=None, dash=None, opacity=None,
                  mode='lines+markers', yaxis=1, fill=None, text='', style='line',
                  markersize=6):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    label : array-like, optional

    Returns
    -------
    Chart

    """
    assert x is not None or y is not None, "x or y must be something"
    if y is None:
        y = x
        x = None
    if x is None:
        x = np.arange(len(y))
    else:
        x = _try_pydatetime(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    assert x.shape[0] == y.shape[0]
    xtype = check_type(x[0])
    ytype = check_type(y[0])

    return Chart(dict(
        xAxis=[
            dict(
                type=xtype,
            )
        ],
        yAxis=[
            dict(
                type=ytype,
            )
        ],
        series=[dict(
            type=style,
            name=name,
            data=np.stack((x, y)).T
        )]
    ))

def line(x=None, y=None, name=None, color=None, width=None, dash=None, opacity=None,
         mode='lines+markers', yaxis=1, fill=None, text="", style='line',
         markersize=6):
    return _simple_chart(
        x=x,
        y=y,
        name=name,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        yaxis=yaxis,
        fill=fill,
        text=text,
        style=style,
        markersize=markersize
    )



def bar(x=None, y=None, name=None, color=None, width=None, dash=None, opacity=None,
        mode='lines+markers', yaxis=1, fill=None, text="", style='bar',
        markersize=6):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    name : array-like, optional

    Returns
    -------
    Chart

    """
    return _simple_chart(
        x=x,
        y=y,
        name=name,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        yaxis=yaxis,
        fill=fill,
        text=text,
        style=style,
        markersize=markersize
    )


def scatter(x=None, y=None, name=None, color=None, width=None, dash=None, opacity=None,
        mode='lines+markers', yaxis=1, fill=None, text="", style='scatter',
        markersize=6):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    name : array-like, optional

    Returns
    -------
    Chart

    """
    return _simple_chart(
        x=x,
        y=y,
        name=name,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        yaxis=yaxis,
        fill=fill,
        text=text,
        style=style,
        markersize=markersize
    )


def heatmap(x=None, y=None, name=None, color=None, width=None, dash=None, opacity=None,
            mode='lines+markers', yaxis=1, fill=None, text="", style='heatmap',
            markersize=6):
    """Draws connected dots.

    Parameters
    ----------
    x : array-like, optional
    y : array-like, optional
    name : array-like, optional

    Returns
    -------
    Chart

    """
    return _simple_chart(
        x=x,
        y=y,
        name=name,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        mode=mode,
        yaxis=yaxis,
        fill=fill,
        text=text,
        style=style,
        markersize=markersize
    )

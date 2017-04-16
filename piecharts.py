"""Echarts python wrapper."""

from __future__ import division

import os
import webbrowser
from tempfile import NamedTemporaryFile
from collections import defaultdict
import json


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
        # self.data = data
        # if data is None:
        #     self.data = []
        # self.layout = layout
        # if layout is None:
        #     layout = {}
        # self.layout = _recursive_dict(layout)
        # self.figure_ = None

    def __add__(self, other):
        self.data += other.data
        self.layout = _merge_layout(self.layout, other.layout)
        return self

    def __radd__(self, other):
        return self.__add__(other)

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

    def legend(self, visible=True):
        """Make legend visible.

        Parameters
        ----------
        visible : bool, optional

        Returns
        -------
        Chart

        """
        self.layout['showlegend'] = visible
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
        self.layout['xaxis']['title'] = label
        return self

    def ylabel(self, label, index=1):
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
        self.layout['yaxis' + str(index)]['title'] = label
        return self

    def zlabel(self, label):
        """Sets the z-axis title.

        Parameters
        ----------
        value : str
            Label for the z-axis

        Returns
        -------
        Chart

        """
        self.layout['zaxis']['title'] = label
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
        self.layout['xaxis']['range'] = [low, high]
        return self

    def ylim(self, low, high, index=1):
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
        self.layout['yaxis' + str(index)]['range'] = [low, high]
        return self

    def xdtick(self, dtick):
        self.layout['xaxis']['dtick'] = dtick
        return self

    def ydtick(self, dtick, index=1):
        self.layout['yaxis' + str(index)]['dtick'] = dtick
        return self

    def xnticks(self, nticks):
        self.layout['xaxis']['nticks'] = nticks
        return self

    def ynticks(self, nticks, index=1):
        self.layout['yaxis' + str(index)]['nticks'] = nticks
        return self

    def yaxis_left(self, index=1):
        """Puts the yaxis on the left hand side

        Parameters
        ----------
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['side'] = 'left'

    def yaxis_right(self, index=1):
        """Puts the yaxis on the right hand side

        Parameters
        ----------
        index : int, optional

        Returns
        -------
        Chart

        """
        self.layout['yaxis' + str(index)]['side'] = 'right'

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
            loader=FileSystemLoader(os.path.dirname(__file__)),
            trim_blocks=True,
            lstrip_blocks=True
        )

        print(jdumps(self.chart))

        html = env.get_template('chart.j2')
        return html.render(
            version=ECHARTS_VERSION,
            opt=jdumps(self.chart)
        )

    def show(self):
        # TODO notebook integration
        with NamedTemporaryFile(prefix='echarts', suffix='.html', mode='w+', delete=False) as f:
            f.write(self._html)
            fname = f.name
        webbrowser.open('file://' + fname)


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

    def save(self, filename=None, show_link=True, auto_open=False,
             output='file', plotlyjs=True):
        pass
        # if filename is None:
        #     filename = NamedTemporaryFile(prefix='plotly', suffix='.html', delete=False).name
        # self.figure_ = go.Figure(data=self.data, layout=go.Layout(**self.layout))
        # # NOTE: this doesn't work for output 'div'
        # py.plot(self.figure_, show_link=show_link, filename=filename, auto_open=auto_open,
        #         output_type=output, include_plotlyjs=plotlyjs)
        # return filename


def line(x=None, y=None, label=None, color=None, width=None, dash=None, opacity=None,
         mode='lines+markers', yaxis=1, fill=None, text="",
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
    # if y.ndim == 2:
    #     if not hasattr(label, '__iter__'):
    #         if label is None:
    #             label = _labels()
    #         else:
    #             label = _labels(label)
    #     data = [go.Scatter(x=x, y=yy, name=ll, line=lineattr, mode=mode, text=text,
    #                        fill=fill, opacity=opacity, yaxis=yn, marker=dict(size=markersize))
    #             for ll, yy in zip(label, y.T)]
    # else:
    #     data = [go.Scatter(x=x, y=y, name=label, line=lineattr, mode=mode, text=text,
    #                        fill=fill, opacity=opacity, yaxis=yn, marker=dict(size=markersize))]
    # if yaxis == 1:
    #     return Chart(data=data)

    return Chart(dict(
        xAxis=[
            dict(
                type='value',
            )
        ],
        yAxis=[
            dict(
                type='value',
            )
        ],
        series=dict(
            type='line',
            data=np.stack((x, y)).T
        )
    ))


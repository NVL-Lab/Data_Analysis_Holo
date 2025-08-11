
__author__ = 'Nuria'

# __author__ = ('Nuria', 'John Doe')

# some utils that may be shared among some plot functions
from typing import Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from utils.analysis_constants import AnalysisConstants


def open_plot(sizes: tuple = (8, 6)):
    """ Function to create a fig with 1 subplot
    :param sizes: size of the figure as a tuple
    :return: a matplotlib figure and the axis of the figure"""
    fig = plt.figure(figsize=sizes)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def open_2subplots():
    """ Function to create a fig with 2 subplot in a row
    :return: a matplotlib figure and the axis of the figure"""
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bx.spines['top'].set_visible(False)
    bx.spines['right'].set_visible(False)
    return fig, ax, bx

def open_4subplots_line():
    """ Function to create a fig with 4 subplot in a row
    :return: a matplotlib figure and the axis of the figure"""
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(141)
    bx = fig.add_subplot(142)
    cx = fig.add_subplot(143)
    dx = fig.add_subplot(144)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    bx.spines['top'].set_visible(False)
    bx.spines['right'].set_visible(False)
    cx.spines['top'].set_visible(False)
    cx.spines['right'].set_visible(False)
    dx.spines['top'].set_visible(False)
    dx.spines['right'].set_visible(False)
    return fig, ax, bx, cx, dx


def open_xsubplots(num_subplots: int = 4):
    """ Function to create a fig with variable number of subplots in a NxN array
    :param num_subplots: number of subplots to create
    :return: a matplotlib figure and the axis of the figure"""
    fig = plt.figure(figsize=(12, 8))
    subplots = []
    for ind in np.arange(1, num_subplots + 1):
        ax = fig.add_subplot(np.ceil(np.sqrt(num_subplots)), np.ceil(np.sqrt(num_subplots)), ind)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        subplots.append(ax)
    return fig, subplots


def save_plot(fig: plt.figure, ax: Optional, folder_path: Path, var_sig: str = '', var_type: str = '',
              set_labels: bool = True):
    """ Function to save a plot to a folder
    :param fig: matplotlib figure
    :param ax: matplotlib axes object
    :param folder_path: folder path to save the figure
    :param var_sig: name of a possible variable to give to the file (used as y_label if set_labels is True)
    :param var_type: name of a possible typle to give to the file (used as x_label if set_labels is True)
    :param set_labels: if True will give the labels of the var_sig var_type"""
    if set_labels:
        ax.set_xlabel(var_type)
        ax.set_ylabel(var_sig)
    file_name = var_sig + '_' + var_type
    file_png = file_name + '.png'
    fig.savefig(folder_path / file_png, bbox_inches='tight')
    file_eps = file_name + '.eps'
    fig.savefig(folder_path / file_eps, format='eps', bbox_inches='tight')
    plt.close(fig)


def easy_plot(arr: np.array, xx: Optional[np.array] = None, folder_plots: Optional[Path] = None,
              var_sig: Optional[str] = None, vertical_array: Optional[np.array] = None):
    """ Function to plot an array with a possible vertical line and save it
    :param arr: array to be plotted
    :param xx: x position of the aa array
    :param folder_plots: folder where to sabe the plot
    :param var_sig: name of a possible variable to give to the file
    :param vertical_array: where to plot a vertical array"""
    fig1, ax1 = open_plot()
    if xx is not None:
        ax1.plot(xx, arr)
    else:
        ax1.plot(arr)
    if vertical_array is not None:
        for vl in vertical_array:
            plt.vlines(x=vl, ymin=np.nanmin(arr), ymax=np.nanmean(arr), color='r')
    if folder_plots is not None:
        if var_sig is None: var_sig = 'kk'
        save_plot(fig1, ax1, folder_plots, var_sig)


def easy_imshow(arr: np.array, folder_plots: Optional[Path] = None, var_sig: Optional[str] = None):
    fig1, ax1 = open_plot()
    """ Function to plot an imshow and save it
    :param arr: 2D array to be plotted
    :param folder_plots: folder where to save the plot
    :param var_sig: name of a possible variable to give to the file"""
    ax1.imshow(arr)
    if folder_plots is not None:
        if var_sig is None: var_sig = 'kk'
        save_plot(fig1, ax1, folder_plots, var_sig)


def get_pvalues(a:np.array, b:np.array, ax, pos: float = 0, height: float = 0.13, ind: bool = True):
    """ Function to calculate the p values based on individual or relative ttest and plot it in a fig
    :param a: First variable to be tested
    :param b: Second variable to be tested
    :param ax: Axes of the plot where to put the result of the test
    :param pos: Position where to put the significance result
    :param height: height where to put the significance result
    :param ind: If True does independent Ttest, if false it does the relative ttest. If relative
    the leng of both a and b has to be the same"""
    if ind:
        _, p_value = stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)])
    else:
        _, p_value = stats.ttest_rel(a, b)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 10, 'p = %0.2E' % p_value)


def get_1s_pvalues(a: np.array, b: float, ax, pos: float = 0, height: float = 0.13):
    """ Function to plot the ttest 1 sample
    :param a: Variable to test
    :param b: Value to be tested agains
    :param ax: Axes of the plot where to put the result
    :param pos: Position where to put the significance result
    :param height: height where to put the significance result"""
    _, p_value = stats.ttest_1samp(a[~np.isnan(a)], b)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 3, 'p = %0.2E' % p_value)


def get_anova_pvalues(a:np.array, b: np.array, axis: int, ax, pos: float = 0, height: float = 0.13):
    """ Function to calculate the p values based on anova plot it in a fig
    :param a: First variable to be tested
    :param b: Second variable to be tested
    :param axis: which axis of a/b will be tested
    :param ax: Axes of the plot where to put the result of the test
    :param pos: Position where to put the significance result
    :param height: height where to put the significance result"""
    _, p_value = stats.f_oneway(a, b, axis=axis)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 3, 'p = %0.2E' % p_value)


def get_reg_pvalues(arr: np.array, x: np.array, ax, pos: float = 0, height: float = 0.13):
    """ Function to calculate the p values based on linear regression plot it in a fig
    :param arr: array to be tested
    :param x: x position/indexes of the array to be tested
    :param ax: Axes of the plot where to put the result of the test
    :param pos: Position where to put the significance result
    :param height: height where to put the significance result"""
    _, _, _, p_value, _ = stats.linregress(x[~np.isnan(arr)], arr[~np.isnan(arr)])
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, 0.9*height, 'p = %0.2E' % p_value)


def calc_pvalue(p_value: float) -> str:
    """ Function that returns a string with the pvalue ready to plot
    :param p_value: the p-value to be plotted
    :return: a string with the significance symbol"""
    if p_value <= 0.001:
        p = '***'
    elif p_value <= 0.01:
        p = '**'
    elif p_value <= 0.05:
        p = '*'
    elif np.isnan(p_value):
        p = 'nan'
    else:
        p = 'ns'
    return p


def generate_palette_all_figures(items: list, palette: str = 'copper') -> dict:
    """ function to generate palette for all elements of an array for all figures
    :param items: list of items (for example mice identities) to have different colors
    :param palette: palette to use
    :return: a dictionary with the different colors for each of the items"""
    custom_palette = sns.color_palette(palette, n_colors=len(items))
    return {index: color for index, color in zip(items, custom_palette)}


def array_regplot(df: pd.DataFrame, column: str) -> Tuple[np.array, np.array]:
    """ function to return x and y values for a regplot
    :param df: dataframe to obtain the x,y values from
    :param column: column to use
    :return: x,y values for a regplot"""
    expanded_values = df[column].apply(pd.Series)
    df_cleaned = expanded_values.dropna(axis=1, how='all')
    array_data = df_cleaned.to_numpy()
    return flatten_array(array_data)


def flatten_array(arr: np.array) -> Tuple[np.array, np.array]:
    """ function to return x and y from a np.array
    :param arr: the array to be flatten
    :return: Tuple with the x and array flattened values"""
    arr_1d = arr.flatten()
    rows, cols = arr.shape
    x = np.tile(np.arange(cols), rows)
    return x, arr_1d


def scale_array(arr: np.array, upper_val: int = 255, lower_val: int = 0) -> np.array:
    """ function to scale an array from lower_val to upper_val
    :param arr: the array to be scaled
    :param upper_val: the upper value of the array
    :param lower_val: the lower value of the array
    :return: the array scaled to the upper and lower val"""
    min_value = arr.min()
    max_value = arr.max()

    scaled_matrix = (arr - min_value) / (max_value - min_value) * (upper_val - lower_val) + lower_val
    return scaled_matrix

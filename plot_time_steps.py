# /usr/bin/env python3
import matplotlib.pyplot as plt


def prettify_variable_names(a_string):
    """
    Converts a variable name to 'prettier' format for presentation in non-code
        contexts

    Specifically handles Python variables named with underscores between words
        by replacing underscores with whitespaces and capitalizing each word
    For example, the string 'variable_number_1' becomes 'Variable Number 1'

    :param a_string: a Python string
    :return: a 'prettified' Python string
    """

    return a_string.replace('_', ' ').title()


def create_data(global_step_n=1000, episodes_n=10, actions_n=4, seed=396711):
    """
    Create data in the same format as might be used in a reinforcement learning
        problem

    :param global_step_n: int, number of time steps across episodes
    :param episodes_n: int, number of episodes
    :param actions_n: int, number of unique actions that agent can execute
    :return:  Pandas DataFrame
    """

    assert episodes_n < global_step_n

    import numpy as np
    import pandas as pd

    np.random.seed(seed)

    df = pd.DataFrame(
        {'global_step': range(global_step_n),
         'state_var01': np.random.normal(0, 1, global_step_n),
         'state_var02': np.random.normal(0, 1, global_step_n),
         'state_var03': np.random.normal(0, 1, global_step_n),
         'action': np.random.randint(0, actions_n, global_step_n),
         'reward': np.random.normal(0, 1, global_step_n),
         })

    # convert integer-encoded actions to names
    action_dict = {
        e: chr(e+77) + chr(e+106) + chr(e+116) for e in df['action'].unique()}
    df['action_name'] = df['action'].map(action_dict)

    # create Boolean 1-D array marking as 'True' which global steps are terminations
    #   of episodes
    episode_termination = np.zeros(global_step_n)
    termination_idx = np.random.choice(
        df['global_step'], size=episodes_n, replace=False)
    episode_termination[termination_idx] = True
    episode_termination = episode_termination > 0
    df['episode_termination'] = episode_termination

    # create index across episodes
    episode_idx = np.array(df['episode_termination'].cumsum())
    episode_idx[termination_idx] = episode_idx[termination_idx] - 1
    df['episode_idx'] = episode_idx

    # create index for steps within each episode
    termination_idx.sort()
    termination_idx = np.append(-1, termination_idx)
    termination_idx = np.append(termination_idx, global_step_n-1)
    episode_lengths = termination_idx[1:] - termination_idx[:-1]
    episode_step_idx_list = [range(e) for e in episode_lengths]
    episode_step_idx = np.concatenate(episode_step_idx_list )
    df['episode_step_idx'] = episode_step_idx

    # calculate cumulative reward for each episode
    df['cumulative_reward'] = df['reward'].groupby(df['episode_idx']).cumsum()

    return df


def plot_groups_of_time_series(
        list_of_dataframes, x_axis_colname, y_axis_colname, group_colname,
        output_filepath=None, plot_group_means=True, palette=None, alpha=0.2,
        title=None, **kwargs):
    """
    Plots groups of time series (i.e., each group may have multiple time
        series) and the means of each group at each time step

    Each group is input as a separate dataframe and is plotted in a different
        color

    Group means (if chosen to be plotted) are plotted as solid, opaque lines,
        whereas individual time series are translucent, based on user-chosen
        'alpha' value

    :param list_of_dataframes: list of Pandas DataFrames, each containing a
        group of time series
    :param x_axis_colname: str, name of dataframe column that contains the time
        step to be plotted on the x-axis
    :param y_axis_colname: str, name of dataframe column that contains the value
        to be plotted on the y-axis
    :param group_colname: str, name of dataframe column that demarcates/indexes
        different time series within each group/dataframe (i.e., episodes)
    :param output_filepath: str, location at which to save the plot
    :param plot_group_means: bool, if 'True', group means are plotted; if
        'False', group means are not plotted
    :param palette: list of strings representing hexadecimal colors used for plotting
        each dataframe in 'list_of_dataframes'
    :param alpha: float ranging from 0 to 1 indicating translucency/opacity of
        individual times series lines
    :param title: str, title printed at top of plot
    :param **kwargs: any additional plotting parameters to be passed to plotting
        package Seaborn's 'lineplot' function
    :return:
    """

    from os import getcwd as os_getcwd
    from os.path import join as os_join
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')

    dfs = list_of_dataframes

    if not output_filepath:
        output_filepath = os_join(os_getcwd(), 'plot.png')

    if not palette:
        palette = sns.color_palette('bright', len(dfs)).as_hex()

    if plot_group_means:
        group_means = [
            dfs[i][y_axis_colname].groupby(dfs[i][x_axis_colname]).mean()
            for i in range(len(dfs))]

    for i, df in enumerate(dfs):

        # plot each time series within a group as the same color; different
        #   groups are plotted in different colors
        df_color = palette[i]
        df_colors = [df_color] * df[group_colname].nunique()

        plot = sns.lineplot(
            x=x_axis_colname, y=y_axis_colname, data=df, hue=group_colname,
            palette=df_colors, alpha=alpha, legend=False, **kwargs)

        if plot_group_means:
            plot_mean = sns.lineplot(
                x=group_means[i].index, y=group_means[i], color=df_colors[0],
                dashes=1, alpha=1, legend=False, label=i)

    if title:
        plt.title(title, fontsize=15)
    else:
        plt.suptitle('legend shows index of dataframe list', fontsize=10)
        plt.title(
            'Darkest line of each color is group mean at each time step',
            fontsize=10)

    x_label = prettify_variable_names(x_axis_colname)
    plt.xlabel(x_label, fontsize=10)
    y_label = prettify_variable_names(y_axis_colname)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(loc='lower right')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_boxplot_stripplot(
        dataframe, x_axis_colname, y_axis_colname, output_filepath=None,
        palette=None, title=None):
    """
    Plots a stripplot atop a boxplot for multiple data distributions

    :param dataframe: Pandas DataFrame
    :param x_axis_colname: str, name of dataframe column containing the values/
        names/indices denoting each data distribution to be plotted
    :param y_axis_colname: str, name of dataframe column containing the values
        to be plotted on the y-axis
    :param output_filepath: str, location at which to save the plot
    :param palette: list of strings representing colors used for plotting
        each data distribution
    :param title: str, title printed at top of plot
    :return:
    """

    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.boxplot(
        x=x_axis_colname, y=y_axis_colname, data=dataframe, palette=palette)
    sns.stripplot(
        x=x_axis_colname, y=y_axis_colname, data=dataframe, linewidth=1,
        palette=palette)

    plt.title(title, fontsize=10)
    x_label = prettify_variable_names(x_axis_colname)
    plt.xlabel(x_label, fontsize=10)
    y_label = prettify_variable_names(y_axis_colname)
    plt.ylabel(y_label, fontsize=10)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_events_by_episode(
        a_dataframe, x_axis_colname, y_axis_colname, event_colname,
        events_list=None, output_filepath=None, palette=None,
        figsize=(6.4, 4.8), title=None, **kwargs):
    """
    Plots events that may occur at each time step of an episode, and plots these
        across multiple episodes

    :param a_dataframe: a Pandas DataFrame
    :param x_axis_colname: str, name of dataframe column that contains the time
        step (which may be a float or integer) to be plotted on the x-axis
    :param y_axis_colname: str, name of dataframe column that contains the index
        of the episode (which is presumed to be an integer but may be a float)
        to be plotted on the y-axis, i.e., so that episodes are separated
        vertically and each time step within an episode appears along a
        horizontal line specified by the episode index
    :param event_colname: str, name of dataframe column that specifies the
        events (which are discrete, e.g., integers or strings) that may occur
        at each time step of an episode
    :param output_filepath: str, location at which to save the plot
    :param palette: list of strings representing hexadecimal colors used for
        plotting each dataframe in 'list_of_dataframes'
    :param figsize: tuple of two floats, specifies width and height of the
        figure for Matplotlib/Seaborn axes
    :param title: str, title printed at top of plot
    :param **kwargs: any additional plotting parameters to be passed to plotting
        package Seaborn's 'scatterplot' function
    :return:
    """

    from os import getcwd as os_getcwd
    from os.path import join as os_join
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')

    if not output_filepath:
        output_filepath = os_join(os_getcwd(), 'plot.png')

    episode_lengths = (
        a_dataframe[y_axis_colname].value_counts().sort_index().reset_index())

    if events_list:
        a_dataframe = a_dataframe.copy()
        a_dataframe = (
            a_dataframe.loc[a_dataframe[event_colname].isin(events_list), :])

    if not palette:
        n_colors = a_dataframe[event_colname].nunique()
        palette = sns.color_palette('bright', n_colors).as_hex()

    _, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=a_dataframe, x=x_axis_colname, y=y_axis_colname, hue=event_colname,
        palette=palette, ax=ax, **kwargs)
    sns.scatterplot(
        data=episode_lengths, x=y_axis_colname, y='index', color='#000000',
        marker='x')

    if title:
        plt.title(title, fontsize=15)
    else:
        plt.suptitle('legend shows different events that occur within an episode', fontsize=10)
        plt.title(
            'X-axis shows each time step within an episode; Y-axis shows different episodes',
            fontsize=10)

    x_label = prettify_variable_names(x_axis_colname)
    plt.xlabel(x_label, fontsize=10)
    y_label = prettify_variable_names(y_axis_colname)
    plt.ylabel(y_label, fontsize=10)
    plt.legend(loc='upper right')

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def main():
    """
    """

    import os
    import pandas as pd
    from numpy import where as np_where
    import seaborn as sns

    pd.set_option('display.max_columns', 12)


    ########################################
    # CREATE DATA
    ########################################

    df_n = 3
    dfs = [
        create_data(global_step_n=1000, episodes_n=10, actions_n=4, seed=9473+i)
        for i in range(df_n)]


    # remove time steps from incomplete episode at end of each dataframe
    ########################################

    last_episode_end_idxs = [
        np_where(e['episode_termination'].values)[0].max() for e in dfs]
    dfs = [e.iloc[:last_episode_end_idxs[i] + 1, :] for i, e in enumerate(dfs)]


    ########################################

    palette = sns.color_palette('bright', len(dfs)).as_hex()
    #palette = sns.color_palette('dark', len(dfs)).as_hex()


    ########################################
    # PLOT CUMULATIVE REWARD BY TRAINED AGENT (DENOTED BY COLOR) BY EPISODE (EACH
    #   EPISODE IS A SEPARATE TIME SERIES) ACROSS TIME STEPS WITHIN EACH EPISODE
    ########################################

    output_filepath = os.path.join(
        os.getcwd(), 'agents_cumul_reward_by_episode_by_step.png')
    plot_groups_of_time_series(
        dfs, 'episode_step_idx', 'cumulative_reward', 'episode_idx',
        output_filepath=output_filepath, plot_group_means=True, palette=palette)


    ########################################
    # PLOT CUMULATIVE REWARD BY TRAINED AGENT (DENOTED BY COLOR) BY EPISODE
    ########################################

    agents_cumulative_rewards_by_episode = [
        e.loc[e['episode_termination'], :].loc[:, 'cumulative_reward']
        for e in dfs]

    df_index_col = 'df_index'
    agents_cumulative_rewards_by_episode = [
        pd.DataFrame(
            {'cumulative_reward': agents_cumulative_rewards_by_episode[i],
             df_index_col: i})
        for i in range(len(agents_cumulative_rewards_by_episode))]

    agents_cumulative_rewards_by_episode = pd.concat(
        agents_cumulative_rewards_by_episode)

    ########################################

    output_filepath = os.path.join(
        os.getcwd(), 'agents_cumul_reward_by_episode.png')
    title = 'Cumulative reward by agent (df index)'
    plot_boxplot_stripplot(
        agents_cumulative_rewards_by_episode, df_index_col, 'cumulative_reward',
        output_filepath=output_filepath, palette=palette, title=title)


    ########################################
    # PLOT TIMELINE OF EVENTS DURING EACH EPISODE
    ########################################

    plot_events_by_episode(
        dfs[0], 'episode_step_idx', 'episode_idx', 'action_name',
        output_filepath='timeline01.png', figsize=(6.4*2, 4.8))

    plot_events_by_episode(
        dfs[1], 'episode_step_idx', 'episode_idx', 'action_name',
        events_list=['Nku', 'Mjt'], output_filepath='timeline02.png',
        figsize=(6.4*2, 4.8))

    plot_events_by_episode(
        dfs[2], 'episode_step_idx', 'episode_idx', 'action_name',
        events_list=['Mjt'], output_filepath='timeline03.png',
        figsize=(6.4*2, 4.8))


if __name__ == '__main__':
    main()
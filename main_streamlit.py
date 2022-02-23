import altair
import streamlit as st
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def compute_rmse(y_actual, y_predicted):
    """
    Function allowing to compute rmse. Basically it is just the mean_squared_error from sklearn
    with argument squared on True.

    Args :

    - y_actual : column containing references
    - y_predicted : column containing experiments

    Return :

    - rmse : root mean squared error between y_actual and y_predicted
    """
    rmse = mean_squared_error(y_actual, y_predicted, squared=True)

    return rmse


@st.cache(allow_output_mutation=True)
def get_data(filename):
    my_data = pd.read_csv(filename)

    return my_data


st.set_page_config(layout="wide")

header = st.container()
sdnn_values = st.container()
intro = st.container()
no_filter = st.container()
filter = st.container()


with header:
    st.title('Results visualisation for HRV collect')
    st.text('The scope of this project is to visualise the results '
            'obtained for HRV calculation. It allows to '
            'display the error committed by the method implemented.')


with sdnn_values:
    st.header('PRV SDNN')
    st.text('Values obtained with this dataset : '
            'https://ieee-dataport.org/open-access/ubfc-phys-2')
    st.text('BVP computed with pyVHR : '
            'https://github.com/phuselab/pyVHR')
    st.text('Pipeline created by ResilEyes Therapeutics : '
            'https://github.com/RESILEYES/bvp_extraction/blob/paul/functions.py')
    # os.chdir('/home/pledes/Bureau/data_tests/streamlit/')
    st.text(os.getcwd())
    st.text(os.path.abspath())
    st.text(os.path.join('../', os.getcwd()))
    # st.text(os.getcwd())
    n_task = st.selectbox('From which task do you want to visualise the results ?', options=['t1', 't2', 't3'],
                          index=0)

width = 10
height = 15

all_dfs_nf = {}
all_dfs_f = {}
num_patients = range(1, 11)
window_sizes = np.arange(10, 35, 5)

for sujet in num_patients:
    all_dfs_nf[f's{sujet}'] = {}
    all_dfs_f[f's{sujet}'] = {}
    path_results = f'results/results_vid_s{sujet}_{n_task}'

    for w in window_sizes:
        all_dfs_nf[f's{sujet}'][f'w{w}'] = get_data(os.path.join(path_results, f'sdnn_{w}_filter_False.csv'))
        all_dfs_f[f's{sujet}'][f'w{w}'] = get_data(os.path.join(path_results, f'sdnn_{w}_filter_True.csv'))


with intro:
    options_s = st.multiselect('Choose one or more subjects to display',  options=all_dfs_nf.keys(),
                               default=all_dfs_nf.keys())
    options_w = st.multiselect('Choose one or more window size to display',  options=all_dfs_nf['s1'].keys(),
                               default=all_dfs_nf['s1'].keys())
    options_f = st.multiselect('Choose whether to display results with frequency filtering',
                               options=['No frequency filtering', 'Frequency filtering'],
                               default=['No frequency filtering', 'Frequency filtering'])


def get_all_rmse(df):
    all_rmses = {}

    for s in df.keys():
        all_rmses[s] = {}

        for window in df[s].keys():
            y = df[s][window]
            num = window.replace('w', '')
            try:
                all_rmses[s][window] = compute_rmse(y[f'exp_{num}'], y[f'ref_{num}'])
            except ValueError:
                y.dropna(inplace=True)
                all_rmses[s][window] = compute_rmse(y[f'exp_{num}'], y[f'ref_{num}'])

    return all_rmses


def write_results(title, df):
    st.title(title)
    st.header('Overview')
    my_df = pd.DataFrame(data=get_all_rmse(df))
    st.dataframe(my_df)
    st.line_chart(my_df)

    # for i in num_patients:
    #     st.write(options_w)
    #     st.write(get_all_rmse(df)[f's{i}'])
    #     st.line_chart(pd.DataFrame(data=[options_w, get_all_rmse(df)[f's{i}']]))
        # ax.plot(options_w, get_all_rmse(df)[f's{i}'].values(), label=f's{i}')
        # ax.legend(loc='upper right')
    # breakpoint()
    # st.pyplot(fig)

    for suj in options_s:
        rmses = []
        st.header(f'Sujet n°{suj[1:]}')
        col1, col2 = st.columns([1, 3])

        for win in options_w:
            col1.subheader(f'Window size = {win[1:]}')
            y = df[suj][win]
            col1.write(y)

            new_y = y.dropna()
            y_pred = new_y.iloc[:, 0]
            y_true = new_y.iloc[:, 1]

            rmse = compute_rmse(y_actual=y_true, y_predicted=y_pred)
            rmses.append(rmse)
            col2.subheader(f'RMSE : {rmse}')
            col2.line_chart(y)

        st.subheader(f'RMSE Mean : {np.mean(rmses)}')
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                    unsafe_allow_html=True)


if 'No frequency filtering' in options_f:
    with no_filter:
        write_results('No frequency filtering', all_dfs_nf)

if 'Frequency filtering' in options_f:
    with filter:
        write_results('Frequency filtering', all_dfs_f)

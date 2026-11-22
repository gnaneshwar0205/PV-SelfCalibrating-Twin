from pathlib import Path
import pandas as pd, streamlit as st, plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='PV Self-Calibrating Digital Twin',page_icon='⚡',layout='wide')
st.markdown('<style>.stApp{background:#F5F8FA;color:#263746}.block-container{padding-top:1.5rem} [data-testid="stMetric"]{background:#fff;border:1px solid #DFE8EA;padding:14px;border-radius:12px} section[data-testid="stSidebar"]{background:#EDF3F4} h1,h2,h3{color:#315A60}</style>',unsafe_allow_html=True)
st.title('PV-fed Boost Converter | Self-Calibrating Digital Twin')
st.caption('Separate ASMC and NN-MPC estimation pipelines · simulation-based monitoring')
choice=st.sidebar.selectbox('Controller pipeline',['ASMC','NN-MPC'])
path=ROOT/'results'/choice.lower().replace('-','')/'twin_log.csv'
# Directory key is nnmpc for NN-MPC
if not path.exists(): st.warning(f'No result log at {path}. Run run_project.py first.'); st.stop()
df=pd.read_csv(path)
last=df.iloc[-1]
a,b,c,d=st.columns(4)
a.metric('PV voltage',f"{last.Vpv:.2f} V"); b.metric('DC bus',f"{last.Vout:.2f} V"); c.metric('Inductor current',f"{last.IL:.2f} A"); d.metric('Duty',f"{last.Duty:.3f}")
st.subheader('Plant vs twin')
for title,actual,twin,ylabel in [('PV voltage','Vpv','Vpv_twin','Voltage (V)'),('Inductor current','IL','IL_twin','Current (A)'),('DC bus voltage','Vout','Vout_twin','Voltage (V)')]:
 f=go.Figure(); f.add_scatter(x=df.time_s,y=df[actual],name='Plant'); f.add_scatter(x=df.time_s,y=df[twin],name='Digital Twin',line=dict(dash='dash')); f.update_layout(title=title,template='plotly_white',height=300,xaxis_title='Time (s)',yaxis_title=ylabel); st.plotly_chart(f,use_container_width=True)
st.subheader('Online parameter estimates')
cols=st.columns(4)
for col,label,key,mul,unit in zip(cols,['Inductance','Input capacitor','Output capacitor','Load current'],['L_est_H','Cin_est_F','Cout_est_F','Iload_est_A'],[1e3,1e6,1e6,1],['mH','μF','μF','A']): col.metric(label,f"{last[key]*mul:.3f} {unit}")
f=go.Figure()
for key,label,mul in [('L_est_H','Estimated L (mH)',1e3),('L_true_H','True L (mH)',1e3)]: f.add_scatter(x=df.time_s,y=df[key]*mul,name=label)
f.update_layout(title='Inductor estimation vs simulator truth (truth is evaluation-only)',template='plotly_white',height=330); st.plotly_chart(f,use_container_width=True)
st.subheader('Twin residuals')
cols=st.columns(3)
for col,key,label,unit in zip(cols,['PV_residual','IL_residual','Vout_residual'],['PV residual RMSE','Inductor residual RMSE','Bus residual RMSE'],['V','A','V']): col.metric(label,f"{(df[key].pow(2).mean()**.5):.4f} {unit}")
st.subheader('Experiment records'); st.dataframe(df.tail(100),use_container_width=True)
st.download_button('Download this run CSV',df.to_csv(index=False),file_name=f'{choice.lower()}_twin_log.csv',mime='text/csv')

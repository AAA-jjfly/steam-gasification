import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

@st.cache_resource
def load_model(model_name):
    if model_name == "H2":
        return pickle.load(open("H21.dat","rb"))
#页面设置
st.set_page_config(
    page_title = "生物质蒸汽气化气体产物预测"
    ,layout = "wide"
    ,initial_sidebar_state = "auto"
)

#侧边栏
st.sidebar.title("功能导航")
st.session_state.date_time = datetime.now()
d = st.sidebar.date_input("日期",st.session_state.date_time.date())
t = st.sidebar.time_input("时间",st.session_state.date_time.time())
st.sidebar.divider()
function_choice = st.sidebar.radio("请选择功能：👇"
                  ,('工况预测', '影响规律预测')
                   )

#主界面
st.title("生物质蒸汽气化气体产物预测")
st.header("",divider="rainbow")

#工况预测界面
if function_choice == "工况预测":
    st.subheader("工况预测",divider="green")
    aim = st.radio("您的预测目标是：👇"
                   ,("产物浓度", "气化效率", "碳转化率")
                   ,horizontal=True)
    model = None
    if aim == "产物浓度":
        model = load_model("H2")
        st.info("当前预测目标: 氢气(H₂)浓度",icon="💡")
    elif aim =="气化效率":
        st.warning("气化效率预测功能开发中，敬请期待！")
    else:
        st.warning("碳转化率预测功能开发中，敬请期待！")

#参数输入
    with st.form("user_input"):
        st.subheader("输入参数",divider="gray")
        col1,col2,col3 = st.columns(3)
        with col1:
            A = st.number_input("灰分含量(A, %)", min_value=0.00, max_value=50.00
                                , value=5.00, step=0.10)
            FC = st.number_input("固定碳含量(FC, %)", min_value=0.00, max_value=30.00
                                , value=25.00, step=0.10)
            V = st.number_input("挥发分含量(V, %)", min_value=45.00, max_value=90.00
                                , value=70.00, step=0.10)
        with col2:
            C = st.number_input("碳元素含量(C, %)", min_value=25.00, max_value=60.00
                                , value=55.00, step=0.10)
            H = st.number_input("氢元素含量(H, %)", min_value=0.00, max_value=10.00
                                , value=5.00, step=0.10)
            O = st.number_input("氧元素含量(O, %)", min_value=15.00, max_value=50.00
                                , value=30.00, step=0.10)
        with col3:
            ER = st.slider("氧气当量比(ER)", min_value=0.00, max_value=0.50
                           , value=0.15, step=0.01)
            T = st.slider("反应温度(T, °C)", min_value=600, max_value=1000
                          , value=800, step=10)
            SB = st.slider("生物质与水蒸气质量比(S/B)", min_value=0.00, max_value=5.00
                           , value=1.00, step=0.10)
#参数提交
        submitted = st.form_submit_button("提交预测", use_container_width=True)
        if submitted and model:
            with st.spinner("预测中，请稍候......"):
                temp_feature = [(A, FC, V, C, H, O, ER, T, SB)]
                data_frame = pd.DataFrame(temp_feature, columns=['A', 'FC', 'V', 'C', 'H', 'O', 'ER', 'T', 'SB'])

#模型预测
                try:
                    new_prediction = model.predict(data_frame)[0]
                    st.success("预测完成！")
                    st.subheader("预测结果", divider="green")
                    st.metric(label="氢气(H₂)浓度", value=f"{new_prediction:.2f}%")
                    #结果解读
                except Exception as e:
                    st.error(f"预测失败：{str(e)}")

#影响规律预测界面
else:
    st.subheader("影响规律预测", divider="green")
    model = load_model("H2")
    if not model:
        st.error("模型加载失败，无法进行预测")
        st.stop()
        
    #预测目标选择
    option1 = st.radio("分析参数：👇"
                       ,("氧气当量比", "反应温度", "水蒸气与生物质质量比")
                       ,horizontal=True
                       ,key='selection'
                      )
    with st.form('law_form'):
        st.subheader("参数设置", divider="gray")

        #固定参数设置
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**固定参数**")
            fixed_A = st.number_input("灰分含量(A, %)", value=5.00, key="fix_A")
            fixed_FC = st.number_input("固定碳含量(FC, %)", value=25.00, key="fix_FC")
            fixed_V = st.number_input("挥发分含量(V, %)", value=70.00, key="fix_V")
        with col2:
            st.markdown("**固定参数(续)**")
            fixed_C = st.number_input("碳元素含量(C, %)", value=55.00, key="fix_C")
            fixed_H = st.number_input("氢元素含量(H, %)", value=5.00, key="fix_H")
            fixed_O = st.number_input("氧元素含量(O, %)", value=30.00, key="fix_O")
        #动态参数设置
        st.markdown("**变化参数范围**")
        if option1 == "氧气当量比":
            fixed_T = st.slider("反应温度(T) - 固定值(°C)", 600, 1000, 800)
            fixed_SB = st.slider("生物质与水蒸气质量比(S/B) - 固定值", 0.00, 5.00, 1.00)
            min_ER, max_ER=st.slider(
                "氧气当量比(ER)变化范围"
                , 0.00, 0.50, (0.10, 0.30), 0.01)
            num_point = st.number_input("数据点数量", min_value=3, max_value=20
                                         , value=5, step=1)
            ers = np.linspace(min_ER, max_ER, num_point)
            params = [(fixed_A, fixed_FC, fixed_V, fixed_C, fixed_H, fixed_O, er, fixed_T, fixed_SB)for er in ers]
            index = ers
        elif option1 == "反应温度":
            fixed_ER = st.slider("氧气当量比(ER) - 固定值", 0.00, 0.50, 0.15)
            fixed_SB = st.slider("生物质与水蒸气质量比(S/B) - 固定值", 0.00, 5.00, 1.00)
            min_T, max_T=st.slider(
                "反应温度(T)变化范围(°C)"
                , 600, 1000, (700, 900), 10)
            num_points = st.number_input("数据点数量", min_value=3, max_value=20
                                         , value=5, step=1)
            ts = np.linspace(min_T, max_T, num_points)
            params = [(fixed_A, fixed_FC, fixed_V, fixed_C, fixed_H, fixed_O, fixed_ER, t, fixed_SB)for t in temps]
            index = ts
        else:
            fixed_ER = st.slider("氧气当量比(ER) - 固定值", 0.00, 0.50, 0.15)
            fixed_T = st.slider("反应温度(T) - 固定值(°C)", 600, 1000, 800)
            min_SB, max_SB=st.slider(
                "生物质与水蒸气质量比(S/B)变化范围"
                , 0.00, 5.00, (0.50, 2.00), 0.10)
            num_point = st.number_input("数据点数量", min_value=3, max_value=20
                                         , value=5, step=1)
            sbs = np.linspace(min_SB, max_SB, num_point)
            params = [(fixed_A, fixed_FC, fixed_V, fixed_C, fixed_H, fixed_O, fixed_ER, fixed_T, sb)for sb in sbs]
            index = sbs
        #参数输入
        submitted_law = st.form_submit_button("开始分析", use_container_width=True)
        if submitted_law and model:
            with st.spinner("分析中，请稍候......"):
                data_frame = pd.DataFrame(params, columns=['A', 'FC', 'V', 'C', 'H', 'O', 'ER', 'T', 'SB'])
                try:
                    prediction = model.predict(data_frame)
                    result_law = pd.DataFrame({
                        "参数值":index
                        , "氢气浓度（%）":prediction
                    }).set_index("参数值")
                    st.success("分析完成！")
                    st.subheader("分析结果", divider="green")
    
                    #图表绘制
                    tab1, tab2 = st.tabs(["📈 趋势图", "📊 数据表"])
                    with tab1:
                        st.line_chart(result_law)
                        st.caption(f"{option1}对氢气浓度的影响趋势")
                    with tab2:
                        st.dataframe(result_law.style.format("{:.2f}")
                                    , use_container_width=True)
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")

#页脚
st.divider()
st.caption(f"最后更新时间: {d} {t}")
    
    
    
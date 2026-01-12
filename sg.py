import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import plotly.graph_objects as go
from io import BytesIO
import warnings
import os
import matplotlib.font_manager

# 获取所有字体信息
font_list = matplotlib.font_manager.fontManager.ttflist

# 提取所有字体的名称（并去重）
all_font_names = sorted(set([f.name for f in font_list]))

# 查找包含中、日、韩（CJK）语言标识的字体
cjk_fonts = []
for font in font_list:
    if hasattr(font, 'name') and font.name:
        # 查找字体名或路径中是否包含常见CJK关键词
        lower_name = font.name.lower()
        if any(key in lower_name for key in ['chinese', 'cjk', 'sc', 'tc', 'jp', 'kr', 'han', 'hei', 'song', 'kai', 'gothic', 'mincho']):
            cjk_fonts.append(font.name)

st.write("### 服务器字体环境诊断")
st.write(f"共发现字体数量: {len(all_font_names)}")
st.write("**可用的CJK（中/日/韩）字体名称:**", set(cjk_fonts) if cjk_fonts else "未找到明确的CJK字体")
st.write("**完整字体列表（前50个）:**", all_font_names[:50])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [
        'Source Han Sans CN', # 开源思源黑体
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'PingFang SC',   # 苹方 (macOS)
        'Hiragino Sans GB',  # 冬青黑体 (macOS)
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'DejaVu Sans',   # 英文字体
        'Arial Unicode MS',  # Unicode 字体
    ],
    'axes.unicode_minus': False,  
    # 字体大小
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})
MODEL_MAPPING = {
    "H2": "H21.dat", 
    "CO": "COF.dat", 
    "CO2": "CO2.dat",
    "CH4": "CH4.dat",
    "H2/CO": "H2CO.dat",
}
@st.cache_resource
def load_model(filename):
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_script_dir, filename)
        if not os.path.exists(full_path):
            st.error(f"❌ 找不到文件。尝试寻找路径: {full_path}")
            return None
        with open(full_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ 加载模型 {filename} 失败: {str(e)}")
        return None
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
                  ,('工况预测', '影响规律预测', 'SHAP解释')
                   )

#主界面
st.title("生物质蒸汽气化气体产物预测")
st.header("",divider="rainbow")
#工况预测界面
if function_choice == "工况预测":
    st.subheader("工况预测",divider="green")
    product_options = list(MODEL_MAPPING.keys())[:4]
    selected_product_name = st.selectbox("请选择具体产物：", product_options)
    model_filename = MODEL_MAPPING.get(selected_product_name)
    if model_filename:
        model = load_model(model_filename)
        if model:
            st.info(f"已加载模型: {model_filename} (预测目标: {selected_product_name})", icon="💡")
        else:
            st.error(f"无法加载模型文件: {model_filename}")
    else:
        st.warning("未找到对应的模型映射，请检查配置。")

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
                try:
                        # 检查模型是否需要 "S/B"
                        if hasattr(model, "feature_names_in_"):
                            model_cols = list(model.feature_names_in_)
                            if "S/B" in model_cols and "SB" in data_frame.columns:
                                data_frame = data_frame.rename(columns={"SB": "S/B"})
#模型预测
                        new_prediction = model.predict(data_frame)
                        if hasattr(new_prediction, 'flatten'):
                            val = new_prediction.flatten()[0]
                        elif isinstance(new_prediction, list):
                            val = new_prediction[0]
                        else:
                            val = new_prediction
                        st.success("预测完成！")
                        st.subheader("预测结果", divider="green")
                        st.metric(label=f"{selected_product_name}", value=f"{val:.4f}")
                except Exception as e:
                    st.error(f"预测失败：{str(e)}")
        
    #数据批量上传
    uploaded_file = st.file_uploader("上传包含批量数据的文件", type=["csv", "xlsx"])
    if uploaded_file is not None and model is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                dataframe = pd.read_csv(uploaded_file)
            else:
                dataframe = pd.read_excel(uploaded_file)
            pred_df = dataframe.copy()
            if hasattr(model, "feature_names_in_"):
                model_cols = list(model.feature_names_in_)
                if "S/B" in model_cols and "SB" in pred_df.columns:
                    pred_df = pred_df.rename(columns={"SB": "S/B"})
                elif "SB" in model_cols and "S/B" in pred_df.columns:
                    pred_df = pred_df.rename(columns={"S/B": "SB"})
        #模型预测
            predictions = model.predict(pred_df)
            dataframe[selected_product_name] = predictions
            st.success(f"批量预测完成！已添加 '{selected_product_name}' 列。")
            st.dataframe(dataframe.head())
        #转换导出格式
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)
            st.download_button(label="下载预测结果"
                            , data=output
                            , file_name="预测结果.xlsx"
                            , mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"批量处理失败：{str(e)}")
    elif uploaded_file is not None and model is None:
        st.warning("请先在上方选择预测目标，以便加载对应的模型。")

#影响规律预测界面
elif function_choice == "影响规律预测":
    st.subheader("影响规律预测", divider="green")
    product_options = list(MODEL_MAPPING.keys())[:4]
    selected_product_name = st.selectbox("请选择具体产物：", product_options)
    model_filename = MODEL_MAPPING.get(selected_product_name)
    if model_filename:
        model = load_model(model_filename)
        if model:
            st.info(f"已加载模型: {model_filename} (预测目标: {selected_product_name})", icon="💡")
        else:
            st.error(f"无法加载模型文件: {model_filename}")
    else:
        st.warning("未找到对应的模型映射，请检查配置。")
    #预测目标选择
    #多因素分析
    on = st.toggle("多因素分析")
    if on:
        st.subheader("多因素分析参数选择")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X轴变量", 
                                 ["氧气当量比", "反应温度", "水蒸气与生物质质量比"],
                                 key='x_axis')
        with col2:
            y_variable = [var for var in ["氧气当量比", "反应温度", "水蒸气与生物质质量比"] if var != x_axis]
            y_axis = st.selectbox("Y轴变量", y_variable, key='y_axis')
    #单因素分析
    else: 
        option1 = st.radio("分析参数：👇"
                       ,("氧气当量比", "反应温度", "水蒸气与生物质质量比")
                       ,horizontal=True
                       ,key='selection'
                          )
    # 参数设置表单
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
        if on:
            col1, col2 = st.columns(2)
            with col1:
                # X轴变量范围
                if x_axis == "氧气当量比":
                    x_min, x_max = st.slider("X: 氧气当量比(ER)范围", 0.00, 0.50, (0.10, 0.30), 0.01, key='x_er')
                    x_points = st.number_input("X轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='x_points')
                elif x_axis == "反应温度":
                    x_min, x_max = st.slider("X: 反应温度(T)范围(°C)", 600, 1000, (700, 900), 10, key='x_temp')
                    x_points = st.number_input("X轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='x_points')
                else:
                    x_min, x_max = st.slider("X: S/B范围", 0.00, 5.00, (0.50, 2.00), 0.10, key='x_sb')
                    x_points = st.number_input("X轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='x_points')
            
            with col2:
                # Y轴变量范围
                if y_axis == "氧气当量比":
                    y_min, y_max = st.slider("Y: 氧气当量比(ER)范围", 0.00, 0.50, (0.10, 0.30), 0.01, key='y_er')
                    y_points = st.number_input("Y轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='y_points')
                elif y_axis == "反应温度":
                    y_min, y_max = st.slider("Y: 反应温度(T)范围(°C)", 600, 1000, (700, 900), 10, key='y_temp')
                    y_points = st.number_input("Y轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='y_points')
                else:  
                    y_min, y_max = st.slider("Y: S/B范围", 0.00, 5.00, (0.50, 2.00), 0.10, key='y_sb')
                    y_points = st.number_input("Y轴数据点数量", min_value=3, max_value=20, value=10, step=1, key='y_points')
            # 第三个参数的设置
            if x_axis != "氧气当量比" and y_axis != "氧气当量比":
                fixed_ER = st.slider("氧气当量比(ER) - 固定值", 0.00, 0.50, 0.15, key='fixed_er')
            if x_axis != "反应温度" and y_axis != "反应温度":
                fixed_T = st.slider("反应温度(T) - 固定值(°C)", 600, 1000, 800, key='fixed_temp')
            if x_axis != "水蒸气与生物质质量比" and y_axis != "水蒸气与生物质质量比":
                fixed_SB = st.slider("生物质与水蒸气质量比(S/B) - 固定值", 0.00, 5.00, 1.00, key='fixed_sb')
            submitted_contour = st.form_submit_button("开始分析", use_container_width=True)
            if submitted_contour and model:
                try:
                    x_values = np.linspace(x_min,x_max,x_points)
                    y_values = np.linspace(y_min,y_max,y_points)
                    X, Y = np.meshgrid(x_values, y_values)
                    predictions = []
                    all_params = []
                    for i in range(len(x_values)):
                        for j in range(len(y_values)):
                            param_dict = {
                                'A': fixed_A,
                                'FC': fixed_FC, 
                                'V': fixed_V,
                                'C': fixed_C,
                                'H': fixed_H,
                                'O': fixed_O}
                            if x_axis == "氧气当量比":
                                param_dict['ER'] = x_values[i]
                            elif x_axis == "反应温度":
                                param_dict['T'] = x_values[i]
                            else:  # S/B
                                param_dict['SB'] = x_values[i]
                            if y_axis == "氧气当量比":
                                param_dict['ER'] = y_values[j]
                            elif y_axis == "反应温度":
                                param_dict['T'] = y_values[j]
                            else:  # S/B
                                param_dict['SB'] = y_values[j]
                            if 'ER' not in param_dict:
                                param_dict['ER'] = fixed_ER
                            if 'T' not in param_dict:
                                param_dict['T'] = fixed_T
                            if 'SB' not in param_dict:
                                param_dict['SB'] = fixed_SB
                            all_params.append([param_dict['A'], param_dict['FC'], param_dict['V'],
                                                param_dict['C'], param_dict['H'], param_dict['O'],
                                                param_dict['ER'], param_dict['T'], param_dict['SB']])
                    data_frame = pd.DataFrame(all_params, columns=['A', 'FC', 'V', 'C', 'H', 'O', 'ER', 'T', 'SB'])
                    submitted = st.form_submit_button("提交预测", use_container_width=True)
                    try:
                            # 检查模型是否需要 "S/B"
                            if hasattr(model, "feature_names_in_"):
                                model_cols = list(model.feature_names_in_)
                                if "S/B" in model_cols and "SB" in data_frame.columns:
                                    data_frame = data_frame.rename(columns={"SB": "S/B"})
                            new_prediction = model.predict(data_frame)
                        #结果解读
                    except Exception as e:
                        st.error(f"预测失败：{str(e)}")
                    # 重塑预测结果为网格格式
                    Z = new_prediction.reshape(len(y_values), len(x_values))
                    st.success("多因素分析完成！")
                    st.subheader("分析结果", divider="green")
                    #绘制等高线图
                    fig = go.Figure(data=go.Contour(
                        z=Z,
                        x=x_values, 
                        y=y_values,
                        colorscale='Viridis',
                        contours=dict(
                            showlabels=True,  
                            labelfont=dict(size=15, color='white')),
                        colorbar=dict(
                            title=f"{selected_product_name}浓度 (%)",
                            titleside="right"
                        ),
                        hovertemplate='<b>%{xaxis.title.text}: %{x:.3f}</b><br>' +
                                    '<b>%{yaxis.title.text}: %{y:.3f}</b><br>' +
                                    '<b>{selected_product_name}浓度: %{z:.2f}%</b><extra></extra>'
                                                    )
                                    )
                    
                    # 设置坐标轴标签
                    x_label = "ER" if x_axis == "氧气当量比" else "T (°C)" if x_axis == "反应温度" else "S/B"
                    y_label = "ER" if y_axis == "氧气当量比" else "T (°C)" if y_axis == "反应温度" else "S/B"
                    
                    # 图表整体布局
                    fig.update_layout(
                        title=f'{selected_product_name}浓度等高线图 ({x_axis} vs {y_axis})',
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        height=600,
                        template='plotly_white'
                    )
                    
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示数据表格
                    st.subheader("数据详情", divider="blue")
                    
                    # 创建展示用的数据框
                    result_df = pd.DataFrame({
                        x_axis: np.repeat(x_values, len(y_values)),
                        y_axis: np.tile(y_values, len(x_values)),
                        f"{selected_product_name}浓度 (%)": new_prediction
                    })
                    
                    st.dataframe(result_df.style.format({x_axis: "{:.2f}", y_axis: "{:.2f}", f"{selected_product_name}浓度 (%)": "{:.2f}"}),
                               use_container_width=True,
                               height=300)
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
        else:
            if option1 == "氧气当量比":
                fixed_T = st.slider("反应温度(T) - 固定值(°C)", 600, 1000, 800)
                fixed_SB = st.slider("生物质与水蒸气质量比(S/B) - 固定值", 0.00, 5.00, 1.00)
                min_ER, max_ER=st.slider(
                    "氧气当量比(ER)变化范围"
                    , 0.00, 0.50, (0.10, 0.30), 0.01)
                num_points = st.number_input("数据点数量", min_value=3, max_value=20
                                             , value=5, step=1)
                ers = np.linspace(min_ER, max_ER, num_points)
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
                params = [(fixed_A, fixed_FC, fixed_V, fixed_C, fixed_H, fixed_O, fixed_ER, t, fixed_SB)for t in ts]
                index = ts
            else:
                fixed_ER = st.slider("氧气当量比(ER) - 固定值", 0.00, 0.50, 0.15)
                fixed_T = st.slider("反应温度(T) - 固定值(°C)", 600, 1000, 800)
                min_SB, max_SB=st.slider(
                    "生物质与水蒸气质量比(S/B)变化范围"
                    , 0.00, 5.00, (0.50, 2.00), 0.10)
                num_points = st.number_input("数据点数量", min_value=3, max_value=20
                                             , value=5, step=1)
                sbs = np.linspace(min_SB, max_SB, num_points)
                params = [(fixed_A, fixed_FC, fixed_V, fixed_C, fixed_H, fixed_O, fixed_ER, fixed_T, sb)for sb in sbs]
                index = sbs
        #参数输入
            submitted_law = st.form_submit_button("开始分析", use_container_width=True)
            if submitted_law and model:
                with st.spinner("分析中，请稍候......"):
                    data_frame = pd.DataFrame(params, columns=['A', 'FC', 'V', 'C', 'H', 'O', 'ER', 'T', 'SB'])
                    try:
                            # 检查模型是否需要 "S/B"
                        if hasattr(model, "feature_names_in_"):
                            model_cols = list(model.feature_names_in_)
                            if "S/B" in model_cols and "SB" in data_frame.columns:
                                data_frame = data_frame.rename(columns={"SB": "S/B"})
                        new_prediction = model.predict(data_frame)
                        result_law = pd.DataFrame({
                            "参数值":index
                            , f"{selected_product_name}浓度（%）":new_prediction
                        }).set_index("参数值")
                        st.session_state['result_law'] = result_law
                        st.success("分析完成！")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")

    # 分析结果展示
    if 'result_law' in st.session_state and not on:
        result_law = st.session_state['result_law']
        st.subheader("分析结果", divider="green")
        default_colors = ["#008000", "#FF0000", "#0000FF", "#FFA500", "#800080", "#00CED1", "#FFD700"]
        if 'color_dict' not in st.session_state:
            st.session_state['color_dict'] = {}
        with st.form("color_form"):
            st.markdown("""
                <div style='background:#f3f6fa;padding:18px 18px 8px 18px;border-radius:12px;border:1px solid #e0e0e0;margin-bottom:10px;'>
                <b style='font-size:17px;'>折线颜色自定义</b>
                <span style='color:#888;font-size:14px;margin-left:10px;'>可分别设置每条线的颜色</span>
            """, unsafe_allow_html=True)
            color_dict = {}
            if isinstance(result_law, pd.DataFrame) and result_law.shape[1] > 1:
                cols = st.columns(len(result_law.columns))
                for i, col in enumerate(result_law.columns):
                    with cols[i]:
                        color = st.color_picker(f"{col}", default_colors[i % len(default_colors)], key=f"color_{col}")
                        color_dict[col] = color
            else:
                color = st.color_picker("折线颜色", "#008000", key="color_single")
                color_dict = {"single": color}
            update_color = st.form_submit_button(" 更新折线颜色", use_container_width=True)
            st.markdown("""</div>""", unsafe_allow_html=True)
            if update_color:
                st.session_state['color_dict'] = color_dict
        tab1, tab2 = st.tabs(["📈 趋势图", "📊 数据表"])
        with tab1:
            color_dict = st.session_state.get('color_dict', {})
            fig = go.Figure()
            if isinstance(result_law, pd.DataFrame) and result_law.shape[1] > 1:
                for col in result_law.columns:
                    fig.add_trace(go.Scatter(
                        x=result_law.index,
                        y=result_law[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(color=color_dict.get(col, default_colors[0]), width=3),
                        marker=dict(size=8, symbol='circle'),
                        hovertemplate=f"<b>{col}</b><br>参数值: %{{x}}<br>{selected_product_name}浓度: %{{y:.2f}}%<extra></extra>"
                    ))
            else:
                if isinstance(result_law, pd.DataFrame):
                    y = result_law.iloc[:,0]
                    name = result_law.columns[0]
                else:
                    y = result_law
                    name = ""
                fig.add_trace(go.Scatter(
                    x=getattr(y, 'index', list(range(len(y)))),
                    y=getattr(y, 'values', y),
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color_dict.get("single", default_colors[0]), width=3),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate=f"<b>{name}</b><br>参数值: %{{x}}<br>{selected_product_name}浓度: %{{y:.2f}}%<extra></extra>"
                ))
            x_label = option1
            if "温度" in option1:
                x_label += " (°C)"
            elif "比" in option1:
                x_label += " (无量纲)"
            y_label = f"{selected_product_name}浓度 (%)"
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='#f7f7fa',
                paper_bgcolor='#f7f7fa',
                font=dict(family="Microsoft YaHei, Arial", size=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(showgrid=True, gridcolor="#e0e0e0", title=x_label),
                yaxis=dict(showgrid=True, gridcolor="#e0e0e0", title=y_label),
                title=dict(text=f"{option1}对{selected_product_name}浓度的影响趋势", x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=20))
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.dataframe(result_law.style.format("{:.2f}"), use_container_width=True)
# SHAP分析界面
else :
    st.subheader("📊 SHAP 分析", divider="green")
    #加载数据
    uploaded_file = st.file_uploader("📂 请上传数据文件 (Excel/CSV)", type=["csv", "xlsx"])
    if not uploaded_file:
        st.info("请先上传数据文件以开始分析")
        st.stop() 
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.success(f"✅ 成功加载数据: {data.shape[0]} 行 × {data.shape[1]} 列")
        with st.expander("📋 查看数据前10行"):
            st.dataframe(data.head(10)) 
    except Exception as e:
        st.error(f"❌ 读取文件失败: {e}")
        st.stop()  
    #列选择
    st.subheader("🛠️ 特征与目标选择")
    all_columns = data.columns.tolist()
    st.write(f"所有列: {all_columns}")
    output_columns = st.multiselect(
        "选择目标列（目标变量）",
        all_columns,
    )
    if not output_columns:
        st.warning("请至少选择一个输出列")
        st.stop()
    input_columns = all_columns[:9]
    st.write(f"**输入特征 ({len(input_columns)}个):** {input_columns}")
    st.write(f"**输出变量 ({len(output_columns)}个):** {output_columns}")
    X = data[input_columns]
    y = data[output_columns] if len(output_columns) > 0 else None
    st.write("### 选择分析模式")
    analysis_mode = st.radio(
        "分析方式",
        ["分析单个输出", "对比所有输出", "综合特征重要性"],
        horizontal=True
    )
    # 警告过滤器
    warnings.filterwarnings('ignore', message="property 'feature_names_in_'")
    # 创建一个安全的解释器，不依赖模型属性
    def safe_explainer(model, X_data):
        try:
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            elif hasattr(model, 'get_booster'): # 针对 XGBoost 原生对象
                try:
                    expected_features = model.get_booster().feature_names
                except:
                    pass
            # 定义一个对齐函数
            def align_data(df):
                df_fixed = df.copy()
                if expected_features and "SB" in expected_features and "S/B" in df_fixed.columns:
                    df_fixed = df_fixed.rename(columns={"S/B": "SB"})
                elif expected_features and "S/B" in expected_features and "SB" in df_fixed.columns:
                    df_fixed = df_fixed.rename(columns={"SB": "S/B"})
                return df_fixed
            X_data_aligned = align_data(X_data)
            # 定义预测函数
            def predict_func(X_input):
                # 格式转换：如果 SHAP 传入的是 Numpy 数组，转回 DataFrame
                if isinstance(X_input, np.ndarray):
                    if X_input.ndim == 1:
                        X_input = X_input.reshape(1, -1)
                    if hasattr(X_data_aligned, 'columns'):
                        X_input = pd.DataFrame(X_input, columns=X_data_aligned.columns)
                elif isinstance(X_input, pd.DataFrame):
                    X_input = align_data(X_input)
                # 调用模型预测
                return model.predict(X_input)
            # 初始化 SHAP 
            background = shap.kmeans(X_data, min(50, len(X_data)))
            explainer = shap.KernelExplainer(predict_func, background)
            return explainer
        except Exception as e:
            st.error(f"创建解释器失败: {str(e)}")
            return None
    if analysis_mode == "分析单个输出":
        # 选择要分析的输出
        selected_output = st.selectbox("选择要分析的输出变量", output_columns)
        model_filename = MODEL_MAPPING.get(selected_output)
        if not model_filename:
            st.error(f"❌ 未找到列 '{selected_output}' 对应的模型文件！")
            st.stop()
        # 加载模型
        with st.spinner(f"正在加载模型: {model_filename} ..."):
            current_model = load_model(model_filename)
        if not current_model:
            st.stop()
        st.success(f"已加载模型: {model_filename}")
        tab1, tab2, tab3 = st.tabs(["全局特征重要性", "单样本解释", "特征依赖分析"])
        with tab1:
            st.subheader(f"📈 {selected_output} - 全局特征重要性分析")
            if st.button("计算特征重要性", key="importance"):
                with st.spinner("正在计算SHAP值..."):
                    try:
                        explainer = safe_explainer(current_model, X)
                        if explainer is None:
                            st.stop()
                        # 计算SHAP值
                        sample_size = st.slider("分析样本数量", 100, len(X), min(200, len(X)))
                        X_sample = X.iloc[:sample_size]
                        shap_values = explainer.shap_values(X_sample)
                        # 绘制特征重要性图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # 处理不同形状的SHAP值
                        if isinstance(shap_values, list):
                            # 分类模型：取第一个类别的SHAP值，或者展示多类别
                            shap_data = shap_values[0]
                        else:
                            # 回归模型：直接使用
                            shap_data = shap_values
                        shap.summary_plot(shap_data, X_sample, plot_type="bar", show=False)
                        plt.title(f"特征重要性 (基于{len(X_sample)}个样本)", fontsize=14)
                        st.pyplot(fig)
                        st.subheader("📋 特征重要性排名")
                        # 计算平均绝对SHAP值
                        if isinstance(shap_values, list):
                            shap_abs_mean = np.abs(shap_values[0]).mean(axis=0)
                        else:
                            shap_abs_mean = np.abs(shap_values).mean(axis=0)
                        importance_df = pd.DataFrame({
                            '特征': input_columns,
                            '平均|SHAP|': shap_abs_mean,
                            '排名': np.argsort(-shap_abs_mean) + 1
                        }).sort_values('平均|SHAP|', ascending=False)
                        st.dataframe(importance_df)
                        st.success("✅ 特征重要性分析完成") 
                    except Exception as e:
                        st.error(f"❌ 分析失败: {e}") 
        with tab2:
            st.subheader(f"🔍 {selected_output} - 单样本预测解释")
            # 选择样本
            sample_idx = st.number_input(
                "选择样本编号",
                min_value=0,
                max_value=len(X)-1,
                value=0,
                help="输入要分析的样本在数据集中的索引"
            )
            
            if st.button("分析该样本", key="single"):
                with st.spinner("正在分析..."):
                    try:
                        # 创建解释器
                        explainer = safe_explainer(current_model, X)
                        if explainer is None:
                            st.error("无法创建SHAP解释器")
                            st.stop()
                        # 获取单个样本
                        sample = X.iloc[sample_idx:sample_idx+1]
                        sample_for_pred = sample.copy()
                        if hasattr(current_model, 'feature_names_in_'):
                            model_cols = list(current_model.feature_names_in_)
                            if "SB" in model_cols and "S/B" in sample_for_pred.columns:
                                sample_for_pred = sample_for_pred.rename(columns={"S/B": "SB"})
                            elif "S/B" in model_cols and "SB" in sample_for_pred.columns:
                                sample_for_pred = sample_for_pred.rename(columns={"SB": "S/B"})
                        # 计算该样本的SHAP值
                        shap_values_single = explainer.shap_values(sample)
                        # 显示样本特征值
                        st.write("### 样本特征值")
                        st.dataframe(sample)
                        # 显示模型预测
                        try:
                            prediction = current_model.predict(sample_for_pred)
                            # 处理格式
                            if hasattr(prediction, 'flatten'):
                                pred_val = prediction.flatten()[0]
                            elif isinstance(prediction, list):
                                pred_val = prediction[0]
                            else:
                                pred_val = prediction
                            st.write(f"### 模型预测值: **{pred_val:.4f}**")
                        except Exception as pred_err:
                            st.warning(f"无法获取直接预测值 (但这不影响下方瀑布图): {pred_err}")
                        # 绘制瀑布图
                        st.write("### SHAP瀑布图")
                        plt.rcParams.update({'font.sans-serif': ['DejaVu Sans']})
                        fig, ax = plt.subplots(figsize=(12, 8))
                        # 使用SHAP的瀑布图
                        if isinstance(shap_values_single, list):
                            # 分类模型：取第一个类别
                            shap_values_for_plot = shap_values_single[0][0]
                            expected_value = explainer.expected_value[0]
                        else:
                            # 回归模型
                            shap_values_for_plot = shap_values_single[0]
                            expected_value = explainer.expected_value
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values_for_plot,
                                base_values=expected_value,
                                data=sample.values[0],
                                feature_names=input_columns
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.rcParams.update({'font.sans-serif': ['SimHei']})
                        st.markdown("""
                            在SHAP瀑布图中，**某些数值很小的正贡献值（如+0.08、+0.11）**，有时会显示为指向左侧而非右侧。
                            这是**可视化显示细节问题**，并不影响实际分析结果的正确性。
                                    """)
                        # 显示SHAP值表格
                        st.write("### SHAP值详情")
                        shap_df = pd.DataFrame({
                            '特征': input_columns,
                            '特征值': sample.values[0],
                            'SHAP值': shap_values_for_plot,
                            '贡献方向': ['增加预测' if x > 0 else '减少预测' for x in shap_values_for_plot]
                        }).sort_values('SHAP值', key=abs, ascending=False)
                        st.dataframe(shap_df)
                    except Exception as e:
                        st.error(f"❌ 分析失败: {e}")
        with tab3:
            st.subheader(f"📊 {selected_output} - 特征依赖分析")
            # 选择要分析的特征
            selected_feature = st.selectbox(
                "选择要分析的特征",
                input_columns,
                key="dependence_feature"
            )
            if st.button("生成依赖图", key="dependence"):
                with st.spinner("正在生成依赖图..."):
                    try:
                        # 创建解释器
                        explainer = safe_explainer(current_model, X)
                        if explainer is None:
                            st.error("无法创建SHAP解释器")
                            st.stop()
                        # 计算SHAP值
                        sample_size = min(500, len(X))
                        X_sample = X.iloc[:sample_size]
                        shap_values = explainer.shap_values(X_sample)
                        if isinstance(shap_values, list):
                            shap_data = shap_values[0]
                        else:
                            shap_data = shap_values
                        # 绘制依赖图
                        plt.close('all') 
                        plt.clf()
                        plt.figure(figsize=(10, 6))
                        try:
                            feature_idx = input_columns.index(selected_feature)
                        except ValueError:
                            st.error(f"特征 {selected_feature} 不在输入列中")
                            st.stop()
                        shap.dependence_plot(
                            feature_idx,
                            shap_data,
                            X_sample,
                            feature_names=input_columns,
                            interaction_index='auto', 
                            show=False,              
                            alpha=0.8                
                        )
                        plt.title(f"{selected_feature} 对 {selected_output} 的影响", fontsize=14)
                        fig = plt.gcf()
                        st.pyplot(fig)
                        # 显示该特征的统计信息
                        st.write("---")
                        st.write(f"**{selected_feature} 统计数据:**")
                        col1, col2, col3, col4 = st.columns(4)
                        feat_data = X[selected_feature]
                        col1.metric("平均值", f"{feat_data.mean():.2f}")
                        col2.metric("标准差", f"{feat_data.std():.2f}")
                        col3.metric("最小值", f"{feat_data.min():.2f}")
                        col4.metric("最大值", f"{feat_data.max():.2f}")
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"❌ 分析失败: {e}")
    elif analysis_mode == "对比所有输出":
        st.write("### 所有输出对比分析")
        # 分析设置
        col1, col2 = st.columns(2)
        with col1:
            max_features = st.slider("显示前N个重要特征", 5, 15, 10)
        with col2:
            sample_size = st.slider("分析样本数", 100, len(X), min(500, len(X)))
        if st.button("开始对比分析"):
            with st.spinner("正在计算所有输出的SHAP值..."):
                try:
                    all_importance = {}
                    X_sample = X.iloc[:sample_size]
                    for output_name in output_columns:
                        fname = MODEL_MAPPING.get(output_name)
                        if not fname:
                            st.warning(f"跳过 {output_name}: 未配置映射")
                            continue
                        st.text(f"正在分析: {output_name} ...")
                        temp_model = load_model(fname)
                        if not temp_model:
                            continue
                        explainer = safe_explainer(temp_model, X)
                        if not explainer: 
                            continue
                        # 计算SHAP值
                        shap_values_output = explainer.shap_values(X_sample)
                        # 计算重要性
                        if isinstance(shap_values_output, list):
                            # 分类模型：通常取第一个类别
                            importance = np.abs(shap_values_output[0]).mean(axis=0)
                        else:
                            # 回归模型：直接计算
                            importance = np.abs(shap_values_output).mean(axis=0)
                        # 保存结果
                        importance_df = pd.DataFrame({
                            '特征': input_columns,
                            '重要性': importance
                        }).sort_values('重要性', ascending=False)
                        all_importance[output_name] = importance_df 
                    if not all_importance:
                        st.error("无法计算任何输出的特征重要性")
                        st.stop()   
                    # 创建对比表格
                    comparison_data = []
                    for output_name, importance_df in all_importance.items():
                        top_features = importance_df.head(max_features)
                        for _, row in top_features.iterrows():
                            comparison_data.append({
                                '输出变量': output_name,
                                '特征': row['特征'],
                                '重要性': row['重要性']
                            })
                    comparison_df = pd.DataFrame(comparison_data)
                    # 绘制热力图
                    st.write("#### 热力图对比")
                    pivot_df = comparison_df.pivot_table(
                        index='特征', 
                        columns='输出变量', 
                        values='重要性',
                        aggfunc='mean'
                    ).fillna(0)
                    # 取重要性最高的前N个特征
                    top_features_overall = pivot_df.mean(axis=1).nlargest(max_features).index
                    pivot_top = pivot_df.loc[top_features_overall]
                    fig1, ax1 = plt.subplots(figsize=(12, 10))
                    im = ax1.imshow(pivot_top.values, cmap='YlOrRd', aspect='auto')
                    ax1.set_xticks(np.arange(len(pivot_top.columns)))
                    ax1.set_yticks(np.arange(len(pivot_top.index)))
                    ax1.set_xticklabels(pivot_top.columns, rotation=45, ha='right')
                    ax1.set_yticklabels(pivot_top.index)
                    plt.colorbar(im, ax=ax1)
                    plt.title('特征在不同输出中的重要性对比', fontsize=16)
                    st.pyplot(fig1)
                    # 显示详细数据
                    st.write("#### 详细数据")
                    for output_name, importance_df in all_importance.items():
                        st.write(f"**{output_name}:**")
                        st.dataframe(importance_df.head(max_features), use_container_width=True)
                    # 验证不同输出的差异
                    if len(all_importance) > 1:
                        st.write("#### 验证结果")
                        # 比较前两个输出的前3个特征
                        output_names = list(all_importance.keys())
                        df1 = all_importance[output_names[0]].head(3)
                        df2 = all_importance[output_names[1]].head(3)
                        
                        st.write(f"**{output_names[0]}** 前3重要特征:")
                        st.write(df1[['特征', '重要性']])
                        st.write(f"**{output_names[1]}** 前3重要特征:")
                        st.write(df2[['特征', '重要性']])
                        
                        # 检查是否相同
                        if df1['特征'].tolist() == df2['特征'].tolist():
                            st.warning("⚠️ 前3重要特征相同，可能仍有问题")
                        else:
                            st.success("✅ 不同输出的重要特征不同，修复成功！")
                    st.success(f"✅ 已对比 {len(all_importance)} 个输出变量")
                except Exception as e:
                    st.error(f"❌ 对比分析失败: {e}")
    else:  # 综合特征重要性
        st.write("### 综合特征重要性分析")
        # 分析设置
        sample_size = st.slider("分析样本数", 100, len(X), min(500, len(X)))
        if st.button("计算综合重要性"):
            with st.spinner("正在计算综合特征重要性..."):
                try:
                    all_importance_arrays = []
                    X_sample = X.iloc[:sample_size]
                    for output_name in output_columns:
                        fname = MODEL_MAPPING.get(output_name)
                        if not fname: 
                            continue
                        temp_model = load_model(fname)
                        if not temp_model: 
                            continue
                        explainer = safe_explainer(temp_model, X)
                        # 计算SHAP值
                        shap_values_output = explainer.shap_values(X_sample)
                        # 计算重要性
                        if isinstance(shap_values_output, list):
                            importance = np.abs(shap_values_output[0]).mean(axis=0)
                        else:
                            importance = np.abs(shap_values_output).mean(axis=0)
                        all_importance_arrays.append(importance)
                    if not all_importance_arrays:
                        st.error("无法计算任何输出的重要性")
                        st.stop()
                    # 计算所有输出的平均重要性
                    combined_importance = np.mean(all_importance_arrays, axis=0)
                    # 创建综合重要性表格
                    combined_df = pd.DataFrame({
                        '特征': input_columns,
                        '综合重要性': combined_importance
                    }).sort_values('综合重要性', ascending=False)
                    # 绘制条形图
                    st.write("#### 综合特征重要性排名")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    # 取前15个特征
                    top_n = min(15, len(combined_df))
                    top_df = combined_df.head(top_n)
                    y_pos = np.arange(len(top_df))
                    ax.barh(y_pos, top_df['综合重要性'])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_df['特征'])
                    ax.invert_yaxis()  # 最高的在顶部
                    ax.set_xlabel('综合重要性')
                    ax.set_title('综合特征重要性排名（所有输出平均）', fontsize=16)
                    plt.tight_layout()
                    st.pyplot(fig)
                    # 显示详细表格
                    st.write("#### 详细排名")
                    st.dataframe(combined_df, use_container_width=True)
                    st.success("✅ 综合特征重要性分析完成")
                except Exception as e:
                    st.error(f"❌ 综合分析失败: {e}")
    with st.expander("⚙️ 高级选项"):
        st.write("### 批量分析选项")
        # 批量生成多个样本的分析
        st.write("**批量样本分析**")
        max_len = len(X)
        start_idx = st.number_input("起始样本", 0, len(X)-10, 0)
        end_idx = st.number_input("结束样本", start_idx+1, len(X)-1, start_idx+5)
        if st.button("批量分析"):
            with st.spinner(f"正在分析样本 {start_idx} 到 {end_idx}..."):
                try:
                    explainer = safe_explainer(current_model, X)
                    if explainer is None:
                        st.error("无法创建SHAP解释器")
                        st.stop()
                    batch_data = X.iloc[start_idx:end_idx+1]
                    # 计算预测值和SHAP值
                    predictions = current_model.predict(batch_data)
                    shap_values_batch = explainer.shap_values(batch_data)
                    if isinstance(shap_values_batch, list):
                        shap_data_batch = shap_values_batch[0]
                    else:
                        shap_data_batch = shap_values_batch
                    # 显示结果表格
                    results = []
                    for i, (idx, row) in enumerate(batch_data.iterrows()):
                        shap_row = shap_data_batch[i]
                        pred_val = predictions[i]
                        if isinstance(pred_val, (np.ndarray, list)):
                            if hasattr(pred_val, 'item'):
                                pred_val = pred_val.item() 
                            else:
                                pred_val = pred_val[0]
                        results.append({
                            '样本ID': idx,
                            '预测值': pred_val,
                            '最大正贡献': input_columns[np.argmax(shap_row)],
                            '最大负贡献': input_columns[np.argmin(shap_row)],
                            '总SHAP绝对值': np.sum(np.abs(shap_row))
                        })
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 批量分析失败: {e}")
    with st.expander("ℹ️ 模型信息"):
        if analysis_mode == "分析单个输出":
            st.write("### 模型详情")
            if 'selected_output' in locals() and 'current_model' in locals():
                model_name = MODEL_MAPPING.get(selected_output, '未知')
                st.write(f"**当前加载模型**: {model_name}")
                st.write(f"**模型类型**: {type(current_model).__name__}")
                # 获取模型参数
                try:
                    st.write("**模型参数**:")
                    if hasattr(current_model, 'get_params'):
                        params = current_model.get_params()
                        params_df = pd.DataFrame(list(params.items()), columns=['参数', '值'])
                        st.dataframe(params_df)
                    else:
                        st.info("该模型对象不支持 get_params() 方法")
                except:
                    st.write("无法获取模型参数详情")
            else:
                st.warning("请先加载模型以查看详情")
        else:
            # 在对比模式下，不显示单个模型详情
            st.write("### 模型详情")
            st.info("💡 当前处于【多模型对比】模式。")
            st.write("在此模式下，系统会循环加载不同模型进行计算，因此无法显示单个模型的详细参数。")
            st.write("如需查看特定模型的参数，请切换回 **'分析单个输出'** 模式。")
        st.divider()
        st.write("**数据信息**:")
        st.write(f"- 输入特征数量: {len(input_columns)}")
        st.write(f"- 样本数量: {len(X)}")
        st.write(f"- 输入特征: {', '.join(input_columns)}")
#页脚
st.divider()
st.caption(f"最后更新时间: {d} {t}")
    
    
    

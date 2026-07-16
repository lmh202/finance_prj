# 优化 Portfolio Builder 页面的前端显示效果

## 目标

对 AURORA 主页面(Portfolio Builder)做几处轻量的视觉优化:指标卡更醒目、持仓表的盈亏红绿着色、区块层次更清晰。不改任何业务逻辑。

## 背景

用户请求"优化 src/portfolio.py 的前端显示效果"。经确认,`src/portfolio.py` 是纯后端模块(CSV 持久化 + 估值计算,无 UI 代码);Portfolio Builder 的全部前端渲染在 `app/app.py`(Streamlit,入口 `streamlit run app/app.py`)。因此本计划只改 `app/app.py`。

页面现状:4 个 `st.metric` 指标卡无边框、持仓表 P/L 列无颜色区分正负、各区块(添加持仓 / 指标 / 表格+图)之间缺少视觉分隔。

## 涉及文件

- `app/app.py` — 唯一要修改的文件,只动展示层代码(约 line 211–245 的 metrics 与 Holdings 表格部分,以及区块间分隔)

## 分步实现

1. **指标卡加边框**:4 个 `st.metric`(line 212–216)加 `border=True` 参数,形成卡片感。
2. **持仓表盈亏着色**:`with left:` 块里(line 228 的 `st.dataframe`),把传入的 `display` 换成 pandas Styler:
   ```python
   def _pnl_color(v):
       if pd.isna(v) or v == 0:
           return ""
       return "color: #1a7f37" if v > 0 else "color: #d1242f"

   styled = display.style.map(_pnl_color, subset=["pnl", "pnl_pct"])
   st.dataframe(styled, hide_index=True, width="stretch", column_config={...原有配置不动...})
   ```
   注意:`column_config` 原样保留(Streamlit 中 column_config 的数字格式与 Styler 的文字颜色可以共存);ProgressColumn 的 weight 列不要动。
3. **区块分隔**:在 valuation 区(metrics 行之前)和 "table + allocation" 区之前各加一个 `st.divider()`,并给 metrics 行上方加一个 `st.subheader("📊 Overview")` 使层次与下方 "Holdings" / "Allocation" 一致。
4. **无价格警告微调**:line 206–209 的 `st.warning` 移到 metrics 之后、表格之前(视觉上先看总览再看告警),文案不变。

## 验收标准

- 页面运行无报错,4 个指标卡带边框显示
- 持仓表中 P/L、P/L % 两列:正值绿色、负值红色、无值不着色;其余列外观与原来一致(格式、Weight 进度条不变)
- 新增 Overview 小标题与两处 divider,页面分区清晰
- `src/portfolio.py` 及其他文件零改动;业务逻辑(计算、存储)零改动

## 验证命令

```bash
python -m py_compile app/app.py     # 语法检查
streamlit run app/app.py            # 手动查看页面效果(加载 sample portfolio 检查表格着色)
```

## 约束

- 只改动"涉及文件"中列出的文件(`app/app.py`),不动无关文件
- 禁止 git commit / git push
- 遵循仓库现有代码风格和命名约定(注释密度低、section 分隔注释风格保持)
- 不新增计划书未提及的依赖

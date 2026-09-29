# 部署与性能

[返回 README](../README.md) · [使用与配置](usage.md)

## 已优化的启动路径

看板使用独立的会话数据上下文，只创建数据源和原有适当性规则。不会加载或训练独立决策接口使用的 XGBoost / SHAP 模型。注册特征名移至无模型依赖的模块，导入数据与模拟样本也不会间接加载预测模型。

`requirements.txt` 只安装看板、数据库驱动和分析需要的依赖；完整决策 / REST 服务另装 `requirements-engine.txt`。Streamlit 固定到已验证的 1.64.0，避免升级后行为意外变化。

首次展示客户散点与雷达；切换因子视角、共同因子页或 AI 因子解读时才加载和拟合因子模型。展开画像贡献明细时才生成瀑布图。共同因子缓存最多 16 组输入；上传的数据与个人 AI 配置仍保留在各自会话中。

这些优化减少应用自身的启动工作，不会消除托管休眠、网络连接、首次依赖安装或浏览器下载资源的耗时。

## 免费平台的选择

以下规则于 2026-09-29 核对，平台政策可能变化。

| 平台 | 当前限制 | 适合用途 |
|---|---|---|
| Streamlit Community Cloud | 无访问 12 小时后休眠；修改依赖会重新安装 | 当前默认演示站，优先保留并优化 |
| Render Free | 无访问 15 分钟后休眠，唤醒约 1 分钟；0.1 CPU / 512 MB | 可作为备用访问线路，不能承诺比当前站点快 |
| Hugging Face Docker Spaces | 官方当前要求付费账号计划才能创建计算型 Space | 暂不作为“零费用”迁移方案 |

原站入口继续使用 `master` 分支的 `ui.py`。不要用持续请求绕过平台休眠限制。需要长期常驻时，应选择明确支持常驻的资源，并先确认费用。

## Docker

容器只运行看板，使用非 root 用户，保留默认跨域与 XSRF 防护，不包含本地密钥、数据库或模型文件。

```bash
docker build -t prudence-engine .
docker run --rm -p 127.0.0.1:8501:8501 prudence-engine
```

浏览器访问 `http://127.0.0.1:8501`。平台可以通过 `PORT` 指定端口；健康检查为 `/_stcore/health`。健康检查只证明 HTTP 服务在线，还需打开页面检查散点、画像与因子视图。

## Render 免费备用站

仓库提供 [render.yaml](../render.yaml)，固定 `plan: free`、新加坡区域、`master` 和 Docker 入口，没有创建付费数据库或磁盘。准备好配置不等于已部署。

在已有 Render 账号中选择 New → Blueprint，使用本项目的公开仓库，核对仍为 Free 后创建。平台分配新的访问地址；验证后再决定是否修改 README 的主入口。首次新建账号、登录、仓库授权由账号持有人完成。此部署不替换 Streamlit 现有站点。

免费实例重启后本地文件不保留。页面上传的数据、参数、API 密钥和 AI 解读按当前会话处理；需要长期数据存储时须另外设计持久化。

## 验收与排查

1. 检查部署日志：安装依赖、服务就绪、页面脚本报错是不同阶段。
2. 如果尚未进入页面就卡在 `share.streamlit.io`，先检查平台访问和网络；应用内缓存不能修复该阶段。
3. 若首次访问慢、同一会话操作快，检查休眠唤醒和冷启动；若每次筛选都慢，再检查计算耗时和数据源请求。
4. 打开页面应看到 120 位模拟客户。点击客户、切换因子视图、展开贡献明细、修改利率并提交；无异常且相关结果更新后才算页面验收通过。

官方依据：[Streamlit 管理与休眠](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)、[按需标签内容](https://docs.streamlit.io/develop/api-reference/layout/st.tabs)、[Streamlit Docker](https://docs.streamlit.io/deploy/tutorials/docker)、[Render Free](https://render.com/docs/free)、[Render Blueprint](https://render.com/docs/blueprint-spec)、[Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-overview)。

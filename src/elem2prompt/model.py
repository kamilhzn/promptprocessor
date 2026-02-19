import xgboost as xgb
import pandas as pd
import joblib as jl
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier


class XGBMultiLabelClassifier:
    def __init__(self, random_state=42):
        self.encoder_dict_x = {}  # 输入特征编码器字典
        self.encoder_dict_y = {}  # 输出标签编码器字典
        self.categorical_cols_x = []  # 输入特征名列表
        self.categorical_cols_y = []  # 输出标签名列表
        self.multi_classifier = None  # 多输出分类器实例
        self.random_state = random_state

    def get_xgb_classifier(self, label_idx, data_df):
        """生成XGBoost分类器的函数"""
        return xgb.XGBClassifier(
            objective="multi:softmax",
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=self.random_state,
            verbosity=0,
        )

    class CustomMultiOutputClassifier(MultiOutputClassifier):
        """自定义多输出分类器"""

        def __init__(self, estimator_func, label_df=None, n_jobs=-1):
            self.estimator_func = estimator_func
            self.label_df = label_df
            placeholder_estimator = xgb.XGBClassifier(verbosity=0)
            super().__init__(estimator=placeholder_estimator, n_jobs=n_jobs)

        def fit(self, X, y, **fit_params):
            self.estimators_ = [self.estimator_func(i, self.label_df) for i in range(y.shape[1])]
            return super().fit(X, y, **fit_params)

    def preprocess_data(self, df):
        """数据预处理：编码输入特征和输出标签"""
        # 处理输入特征
        x = df.iloc[:, 0:16].copy()
        self.categorical_cols_x = x.columns.tolist()
        for col in self.categorical_cols_x:
            label_encoder = LabelEncoder()
            x[col] = label_encoder.fit_transform(x[col])
            self.encoder_dict_x[col] = label_encoder

        # 处理输出标签
        y = df.iloc[:, 17:42].copy()
        self.categorical_cols_y = y.columns.tolist()
        for col in self.categorical_cols_y:
            label_encoder = LabelEncoder()
            y[col] = label_encoder.fit_transform(y[col])
            self.encoder_dict_y[col] = label_encoder

        return x, y

    # def train(self, data_path="data.csv", test_size=0.2):
    #     """训练模型"""
    #     df = pd.read_csv(data_path, header=1)

    #     x, y = self.preprocess_data(df)

    #     x_train, x_test, y_train, y_test = train_test_split(
    #         x, y, test_size=test_size, random_state=self.random_state
    #     )

    #     # 初始化并训练多输出分类器
    #     self.multi_classifier = self.CustomMultiOutputClassifier(
    #         estimator_func=self.get_xgb_classifier,
    #         label_df=df,
    #         n_jobs=1
    #     )
    #     self.multi_classifier.fit(x_train, y_train)

    #     # 评估模型
    #     score = self.multi_classifier.score(x_test, y_test)
    #     print(f"模型得分: {score}")
    #     return score

    def predict(self, x_predict, return_description=True):
        """预测函数：接收原始输入特征dataframe，返回解码后的预测结果，返回dataframe"""
        x_pred = x_predict.copy()

        # 编码输入特征
        for col in x_pred.columns:
            if col in self.encoder_dict_x:
                x_pred[col] = self.encoder_dict_x[col].transform(x_pred[col])

        result = self.multi_classifier.predict(x_pred)
        # 解码预测结果
        result_df = pd.DataFrame(result, columns=self.categorical_cols_y)
        for col in result_df.columns:
            result_df[col] = self.encoder_dict_y[col].inverse_transform(result_df[col])
        if return_description:
            return self.df_row_to_description(result_df.iloc[0])
        else:
            return result_df

    # def save_model(self, name='multi_classifier.pkl'):
    #     """保存模型"""
    #     jl.dump(self, open(name, 'wb'))
    #     print(f"模型拥有保存: {name}")

    def predict_comfyui(self, input_str):
        """
        参数:
            input_str: 输入的完整字符串，格式如 '"主色相": ["棕"],"主饱和度": ["低"],...'

        返回:
            格式化后的DataFrame
        """
        clean_str = input_str.strip().strip(",").strip()
        temp_str = clean_str.replace('"],["', '"]###["')
        pairs = [p.strip() for p in temp_str.split("],") if p.strip()]
        data_dict = {}
        for pair in pairs:
            if ":" in pair:
                key_part, value_part = pair.split(":", 1)
            elif "：" in pair:
                key_part, value_part = pair.split("：", 1)
            else:
                continue

            key = key_part.strip().strip('"').strip("'").strip()

            value_clean = value_part.strip().replace("###", ",").strip("[").strip("]").strip('"').strip("'").strip()

            value = ",".join([v.strip().strip('"').strip("'") for v in value_clean.split(",") if v.strip()])

            data_dict[key] = [value]

        df = pd.DataFrame(data_dict)
        return self.predict(df)

    def df_row_to_description(self, row):
        """
        将DataFrame的一行人物特征数据转换为简单的自然描述字符串
        """
        # 过滤词
        exclude_words = ["无"]

        description_parts = []

        # 基础特征
        basic_info = f"一位{row['性别']}，整体画风风格为{row['风格']}，"
        description_parts.append(basic_info)

        # 服饰部分
        clothes_parts = []
        if not any(word in row["上衣"] for word in exclude_words):
            clothes_parts.append(f"上身穿着{row['上衣']}")
        if not any(word in row["下衣"] for word in exclude_words):
            clothes_parts.append(f"下身搭配{row['下衣']}")
        if not any(word in row["鞋袜"] for word in exclude_words):
            clothes_parts.append(f"脚穿{row['鞋袜']}")
        if not any(word in row["头饰"] for word in exclude_words):
            clothes_parts.append(f"头上戴有{row['头饰']}，")

        if clothes_parts:
            description_parts.append("，".join(clothes_parts))

        # 外貌
        appearance_parts = []
        if not any(word in row["头发长度"] for word in exclude_words):
            appearance_parts.append(f"{row['头发长度']}，是{row['头发颜色']}{row['发型']}")
        if not any(word in row["眼睛颜色"] for word in exclude_words):
            appearance_parts.append(f"{row['眼睛颜色']}色眼睛，神态为{row['眼睛状态']}")
        if not any(word in row["皮肤细节"] for word in exclude_words):
            appearance_parts.append(f"面部有{row['皮肤细节']}")
        if appearance_parts:
            description_parts.append(f"拥有{','.join(appearance_parts)}")

        # 动作场景
        action_parts = []
        if not any(word in row["情绪"] for word in exclude_words):
            action_parts.append(f"神情{row['情绪']}")
        if not any(word in row["武器"] for word in exclude_words):
            action_parts.append(f"手持{row['武器']}")
        if not any(word in row["动作"] for word in exclude_words):
            action_parts.append(f"身处{row['背景']}，整体动作为{row['动作']}")
        if not any(word in row["细节动作"] for word in exclude_words):
            action_parts.append(f"细节动作为{row['细节动作']}")
        if action_parts:
            description_parts.append(f"，{','.join(action_parts)}")

        # 细节
        design_parts = []
        if not any(word in row["人物比例"] for word in exclude_words):
            design_parts.append(f"人物比例为{row['人物比例']}")
        if not any(word in row["条纹印花"] for word in exclude_words):
            design_parts.append(f"服饰有{row['条纹印花']}")
        if not any(word in row["材质与气质"] for word in exclude_words):
            design_parts.append(f"材质呈现{row['材质与气质']}")
        if design_parts:
            description_parts.append(f"，设计上{','.join(design_parts)}，衣服主色为{row['衣服颜色']}色")

        # 拼接
        description = "".join(description_parts) + "。"
        return description

    @staticmethod
    def load_model(load_path="multi_classifier.pkl"):
        """加载模型"""
        model = jl.load(open(load_path, "rb"))
        print(f"模型已从: {load_path} 加载")
        return model

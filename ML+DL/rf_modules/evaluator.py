import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class RFEvaluator:
    def __init__(self, label_encoder=None):
        self.label_encoder = label_encoder
    
    def evaluate_model(self, rf_model, X_test, y_test_encoded, feature_type):
        """评估模型性能"""
        print(f"\n开始评估{feature_type}模型...")
        
        # 预测
        start_time = time.time()
        try:
            y_pred = rf_model.predict(X_test)
            prediction_time = time.time() - start_time
        except Exception as e:
            print(f"预测过程中出错: {e}")
            return None
        
        # 确保预测结果是一维数组
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        if y_test_encoded.ndim > 1:
            y_test_encoded = y_test_encoded.flatten()
        
        # 计算基本指标
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        avg_prediction_time = prediction_time / len(y_test_encoded)
        
        print(f"\n{feature_type}随机森林模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"测试预测时间: {prediction_time:.4f}秒")
        print(f"平均每样本预测时间: {avg_prediction_time:.6f}秒")
        
        # 获取类别信息
        all_train_classes = np.unique(rf_model.classes_) if hasattr(rf_model, 'classes_') else np.unique(y_test_encoded)
        all_test_classes = np.unique(y_test_encoded)
        all_pred_classes = np.unique(y_pred)
        all_possible_classes = np.unique(np.concatenate([all_train_classes, all_test_classes, all_pred_classes]))
        
        print(f"模型训练类别: {all_train_classes}")
        print(f"测试集类别: {all_test_classes}")
        print(f"预测类别: {all_pred_classes}")
        
        # 创建target_names
        target_names = self._create_target_names(all_possible_classes)
        
        # 生成分类报告
        report = self._generate_classification_report(y_test_encoded, y_pred, all_possible_classes, target_names)
        
        # 生成混淆矩阵
        cm = self._generate_confusion_matrix(y_test_encoded, y_pred, all_possible_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time,
            'avg_prediction_time': avg_prediction_time,
            'classification_report': report,
            'confusion_matrix': cm,  # 保持为numpy数组，不转换为list
            'target_names': target_names,
            'feature_type': feature_type
        }
    
    def _create_target_names(self, all_possible_classes):
        """创建类别名称"""
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            target_names = []
            for class_idx in all_possible_classes:
                if class_idx < len(self.label_encoder.classes_):
                    target_names.append(self.label_encoder.classes_[class_idx])
                else:
                    target_names.append(f"Class_{class_idx}")
        else:
            target_names = [f"Class_{i}" for i in all_possible_classes]
        
        return target_names
    
    def _generate_classification_report(self, y_test_encoded, y_pred, all_possible_classes, target_names):
        """生成分类报告"""
        try:
            report = classification_report(
                y_test_encoded, y_pred,
                labels=all_possible_classes,
                target_names=target_names,
                zero_division=0,
                output_dict=False
            )
            print(f"\n分类报告:\n{report}")
            return report
        except Exception as e:
            print(f"生成分类报告时出错: {e}")
            try:
                report = classification_report(y_test_encoded, y_pred, zero_division=0)
                print(f"\n简化分类报告:\n{report}")
                return report
            except Exception as e2:
                print(f"生成简化分类报告也失败: {e2}")
                return "分类报告生成失败"
    
    def _generate_confusion_matrix(self, y_test_encoded, y_pred, all_possible_classes):
        """生成混淆矩阵"""
        try:
            cm = confusion_matrix(y_test_encoded, y_pred, labels=all_possible_classes)
            print(f"混淆矩阵形状: {cm.shape}")
            return cm
        except Exception as e:
            print(f"生成混淆矩阵时出错: {e}")
            try:
                cm = confusion_matrix(y_test_encoded, y_pred)
                print(f"使用默认混淆矩阵，形状: {cm.shape}")
                return cm
            except Exception as e2:
                print(f"生成默认混淆矩阵也失败: {e2}")
                return None
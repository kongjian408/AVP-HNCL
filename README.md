# AVP-HNCL

**A Two-Stage Model for AVP Classification**

## abstract
Viral infections have long been a core focus in the field of public health. Antiviral peptides (AVPs), due to their unique mechanisms of action and significant inhibitory effects against a wide range of viruses, exhibit tremendous potential in protecting organisms from various viral diseases. However, existing studies on antiviral peptide recognition often rely on feature selection. As data volume continues to grow and task complexity increases, traditional methods are increasingly showing limitations in feature extraction capabilities and model generalization performance. To tackle these challenges, we propose an innovative two-stage predictive framework that integrates the ESM2 model, data augmentation, feature fusion, and contrastive learning techniques. This framework enables the simultaneous identification of AVPs and their subclasses. By introducing a novel top-k queue-based contrastive learning strategy, the framework significantly improves the model’s accuracy in distinguishing challenging positive and negative samples, as well as its generalization performance. This approach provides robust theoretical support and technical tools for advancing research on antiviral peptides. Model evaluation results show that on the Set 1, the framework achieves an accuracy of 0.9362 and a Matthews Correlation Coefficient (MCC) score of 0.8730. On the Set 2, the model achieves perfect accuracy (1.000) and an MCC score of 1.000. In addition, during the second stage, the model accurately predicts the antiviral activity of antiviral peptides against six major virus families and eight specific viruses. To further enhance accessibility for users, we have developed a user-friendly web interface, available at http://www.bioai-lab.com/AVP-HNCL.

## Conclusion

The current negative sample selection strategy has some limitations and significant room for improvement. For example, while the model can effectively distinguish negative samples highly similar to anchor sequences, during contrastive learning, some of these negative samples may already be clearly differentiated from the anchor samples, yet still get repeatedly used due to their high similarity. This phenomenon not only wastes computational resources but also limits the model’s potential for further optimization.

In the future, we plan to introduce negative sample selection methods based on attention mechanisms and uncertainty evaluation. By dynamically adjusting the importance weights of negative samples, we aim to prioritize those with higher learning value, thus improving training efficiency, avoiding resource waste, and further enhancing the model's learning capacity.




```
Data\raw_data
```



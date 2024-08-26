from transformers_interpret import SequenceClassificationExplainer

def interpret_model(model, text, tokenizer, visualizer = False, id_num=0):
    cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer

    ## This is supposed to output custom lables
    # attribution_type: str = "lig",
    # custom_labels: List[str] | None = None
    )
    # word_attributions = cls_explainer(text)
    # if visualizer:
    #     cls_explainer.visualize(f"/projekte/tcl/users/keithan/projectcalderon/wp1-semantic-analysis/gender-predict-pkg/results/viz_files/{id_num}viz.html")
    # # return word_attributions

    print("Interpreting input text ")

### Try to print all the visualizations into a single file 
    for sentence in text:
       word_attributions = cls_explainer(sentence[:511])
    if visualizer:
        cls_explainer.visualize("calderon-gender-prediction/results/viz_files/viz.html")
    return word_attributions
    
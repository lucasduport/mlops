import joblib
import subprocess

def transpile_model_to_c(model_path, output_c_file, test_features=None):
    """
    Load a joblib model and transpile it to C code
    
    Args:
        model_path: Path to the .joblib file containing the trained model
        output_c_file: Path for the output .c file
        test_features: Optional test features array for the main function
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Detect model type
    model_type = type(model).__name__
    print(f"Model type: {model_type}")
    
    # Handle different model types
    if model_type == 'LogisticRegression':
        is_logistic = True
        is_tree = False
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
        n_features = len(coefficients)
        tree_data = None
    elif model_type == 'LinearRegression':
        is_logistic = False
        is_tree = False
        coefficients = model.coef_
        intercept = model.intercept_
        n_features = len(coefficients)
        tree_data = None
    elif model_type in ['DecisionTreeRegressor', 'DecisionTreeClassifier']:
        is_logistic = False
        is_tree = True
        is_classifier = (model_type == 'DecisionTreeClassifier')
        coefficients = None
        intercept = None
        n_features = model.n_features_in_
        tree_data = {
            'tree': model.tree_,
            'n_features': n_features,
            'is_classifier': is_classifier,
            'n_classes': model.n_classes_ if is_classifier else None
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: LinearRegression, LogisticRegression, DecisionTreeRegressor, DecisionTreeClassifier.")
    
    print("Model loaded successfully!")
    if is_tree:
        print(f"Model type: Decision Tree {'Classifier' if tree_data['is_classifier'] else 'Regressor'}")
        print(f"Number of features: {n_features}")
        print(f"Number of nodes: {tree_data['tree'].node_count}")
        print(f"Max depth: {tree_data['tree'].max_depth}")
    else:
        print(f"Model type: {'Logistic Regression' if is_logistic else 'Linear Regression'}")
        print(f"Number of features: {n_features}")
        print(f"Coefficients: {coefficients}")
        print(f"Intercept: {intercept}")
    
    if test_features is None:
        test_features = [205.9991686803, 2, 0]  # size, nb_rooms, garden
    
    if is_tree:
        c_code = generate_tree_c_code(tree_data, test_features)
    else:
        c_code = generate_c_code(coefficients, intercept, n_features, test_features, is_logistic)

    with open(output_c_file, 'w') as f:
        f.write(c_code)
    
    print(f"\nC code saved to {output_c_file}")
    
    python_prediction = model.predict([test_features])[0]
    
    # For logistic regression, also show probability
    if is_logistic:
        python_proba = model.predict_proba([test_features])[0]
        print(f"\nPython model prediction for {test_features}:")
        print(f"  Class: {python_prediction}")
        print(f"  Probabilities: {python_proba}")
    elif is_tree and tree_data['is_classifier']:
        print(f"\nPython model prediction for {test_features}:")
        print(f"  Class: {python_prediction}")
    else:
        print(f"\nPython model prediction for {test_features}: {python_prediction}")
    
    return output_c_file, python_prediction, is_logistic, is_tree


def generate_c_code(coefficients, intercept, n_features, test_features, is_logistic=False):
    """
    Generate C code for the linear or logistic regression model
    """
    coef_str = ", ".join([f"{coef:.10f}" for coef in coefficients])
    
    test_features_str = ", ".join([f"{feat:.10f}" for feat in test_features])
    
    # Add sigmoid function for logistic regression
    sigmoid_function = """
/**
 * Sigmoid function for logistic regression
 * sigmoid(x) = 1 / (1 + exp(-x))
 */
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}
""" if is_logistic else ""
    
    # Modify prediction function based on model type
    prediction_logic = """    // Calculate prediction: y = intercept + sum(coef_i * feature_i)
    float result = intercept;
    
    for (int i = 0; i < n_features; i++) {
        result += coefficients[i] * features[i];
    }
"""
    
    if is_logistic:
        prediction_logic += """    
    // Apply sigmoid function for logistic regression
    result = sigmoid(result);
"""
    
    return_comment = "probability (0-1)" if is_logistic else "predicted value"
    model_type_comment = "Logistic" if is_logistic else "Linear"
    
    c_code = f"""#include <stdio.h>
#include <math.h>
{sigmoid_function}
/**
 * {model_type_comment} Regression Prediction Function
 * 
 * @param features: Array of input features
 * @param n_features: Number of features
 * @return {return_comment}
 */
float prediction(float *features, int n_features) {{
    float coefficients[{n_features}] = {{{coef_str}}};
    float intercept = {intercept:.10f};
{prediction_logic}    
    return result;
}}

int main() {{
    float test_features[{n_features}] = {{{test_features_str}}};
    float pred = prediction(test_features, {n_features});
    
    printf("Input features: ");
    for (int i = 0; i < {n_features}; i++) {{
        printf("%.4f", test_features[i]);
        if (i < {n_features} - 1) {{
            printf(", ");
        }}
    }}
    printf("\\n");
    """
    
    if is_logistic:
        c_code += """
    printf("Predicted probability: %.6f\\n", pred);
    printf("Predicted class: %d\\n", pred >= 0.5 ? 1 : 0);
"""
    else:
        c_code += """
    printf("Predicted price: %.2f\\n", pred);
"""
    
    c_code += """    
    return 0;
}
"""
    
    return c_code


def generate_tree_c_code(tree_data, test_features):
    """
    Generate C code for decision tree models
    """
    tree = tree_data['tree']
    n_features = tree_data['n_features']
    is_classifier = tree_data['is_classifier']
    
    test_features_str = ", ".join([f"{feat:.10f}" for feat in test_features])
    
    # Generate the tree traversal code
    tree_code = generate_tree_nodes(tree, 0, 1)
    
    model_type = "Decision Tree Classifier" if is_classifier else "Decision Tree Regressor"
    return_type = "class" if is_classifier else "value"
    
    c_code = f"""#include <stdio.h>

/**
 * {model_type} Prediction Function
 * 
 * @param features: Array of input features
 * @return predicted {return_type}
 */
float prediction(float *features) {{
{tree_code}
}}

int main() {{
    float test_features[{n_features}] = {{{test_features_str}}};
    float pred = prediction(test_features);
    
    printf("Input features: ");
    for (int i = 0; i < {n_features}; i++) {{
        printf("%.4f", test_features[i]);
        if (i < {n_features} - 1) {{
            printf(", ");
        }}
    }}
    printf("\\n");
    """
    
    if is_classifier:
        c_code += """
    printf("Predicted class: %d\\n", (int)pred);
"""
    else:
        c_code += """
    printf("Predicted value: %.2f\\n", pred);
"""
    
    c_code += """    
    return 0;
}
"""
    
    return c_code


def generate_tree_nodes(tree, node_id, indent_level):
    """
    Recursively generate C code for tree nodes
    """
    indent = "    " * indent_level
    
    # Check if leaf node
    if tree.feature[node_id] == -2:  # -2 indicates a leaf node in sklearn
        # Leaf node - return the value
        value = tree.value[node_id][0]
        if len(value) > 1:
            # For classification, return the class with max samples
            predicted_class = value.argmax()
            return f"{indent}return {predicted_class}.0;\n"
        else:
            # For regression, return the value
            return f"{indent}return {value[0]:.10f};\n"
    
    # Internal node - generate if-else structure
    feature_idx = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    code = f"{indent}if (features[{feature_idx}] <= {threshold:.10f}) {{\n"
    code += generate_tree_nodes(tree, left_child, indent_level + 1)
    code += f"{indent}}} else {{\n"
    code += generate_tree_nodes(tree, right_child, indent_level + 1)
    code += f"{indent}}}\n"
    
    return code



def compile_and_run(c_file, executable_name="model_prediction"):
    """
    Compile the C file and run it
    """
    compile_cmd = f"gcc -o {executable_name} {c_file} -lm"
    print(f"\nCompilation command: {compile_cmd}")
    
    try:
        print("\nCompiling...")
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Compilation failed!")
            print(f"Error: {result.stderr}")
            return None
        
        print("Compilation successful!")
        
        run_cmd = f"./{executable_name}"
        print(f"\nRunning: {run_cmd}")
        print("-" * 60)
        
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Execution failed!")
            print(f"Error: {result.stderr}")
            return None
        
        print(result.stdout)
        print("-" * 60)
        
        return result.stdout
        
    except Exception as e:
        print(f"Error during compilation/execution: {e}")
        return None


def main():
    model_path = "regression.joblib"
    output_c_file = "model_prediction.c"
    executable_name = "model_prediction"
    
    test_features = [205.9991686803, 2, 0]
    
    c_file, python_prediction, is_logistic, is_tree = transpile_model_to_c(
        model_path=model_path,
        output_c_file=output_c_file,
        test_features=test_features
    )
    
    c_output = compile_and_run(c_file, executable_name)
    
    # Verify predictions match
    if c_output:
        if is_logistic:
            print(f"Python prediction: {python_prediction}")
            
            for line in c_output.split('\n'):
                if 'Predicted class:' in line:
                    c_prediction = int(line.split(':')[1].strip())
                    print(f"C prediction:      {c_prediction}")
                    
                    if int(python_prediction) == c_prediction:
                        print("\nOK - Classifications match!")
                    else:
                        print("\nKO - Classifications differ!")
        elif is_tree:
            print(f"Python prediction: {python_prediction}")
            
            for line in c_output.split('\n'):
                if 'Predicted class:' in line or 'Predicted value:' in line:
                    c_prediction_str = line.split(':')[1].strip()
                    c_prediction = float(c_prediction_str)
                    print(f"C prediction:      {c_prediction}")
                    
                    diff = abs(float(python_prediction) - c_prediction)
                    
                    if diff < 0.01:
                        print("\nOK - Predictions match!")
                    else:
                        print(f"\nDifference: {diff:.6f}")
                        print("KO - Predictions differ!")
        else:
            print(f"Python prediction: {python_prediction:.2f}")
            
            for line in c_output.split('\n'):
                if 'Predicted price:' in line:
                    c_prediction = float(line.split(':')[1].strip())
                    print(f"C prediction:      {c_prediction:.2f}")
                    
                    diff = abs(python_prediction - c_prediction)
                    print(f"\nDifference: {diff:.6f}")
                    
                    if diff < 0.01:
                        print("OK")
                    else:
                        print("KO")
    print("To compile manually, run:")
    print(f"  gcc -o {executable_name} {output_c_file} -lm")
    print("\nTo execute, run:")
    print(f"  ./{executable_name}")

if __name__ == "__main__":
    main()

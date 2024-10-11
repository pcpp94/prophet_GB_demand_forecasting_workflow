### Key Design Considerations

1. **End-to-End ML Lifecycle Management:**
- Machine learning projects involve a variety of stages: data preparation, model training, experiment tracking, packaging, deployment, and monitoring.
- Decision: Create distinct components (e.g., Tracking, Projects, Models, and Model Registry) to address each lifecycle stage, enabling modular usage depending on user needs.

2. **Modularity and Flexibility:**
- Practitioners often use different tools and frameworks, such as Scikit-Learn, TensorFlow, PyTorch, etc. MLflow needed to be framework-agnostic to support these.
- Decision: Design MLflow as a modular framework where each component can be used independently or together, allowing users to pick and choose which parts fit their workflow. For instance, a user might only want to track experiments without using the model packaging or registry components.

3. **Reproducibility and Experiment Tracking:**
- ML experiments are iterative, and keeping track of parameters, metrics, and code versions is crucial for reproducibility and improvement.
- Decision: Create the Tracking component, with classes for Experiment and Run, to manage experiments in a way that logs all key data for each training session.

4. **Standardization of ML Model Packaging and Deployment:**
- The Models component has a standard format for packaging models (the MLmodel file) that can handle various ML frameworks using “flavors” (e.g., Scikit-Learn, PyTorch).
- This structure allows models to be packaged in a uniform way and deployed seamlessly across different environments, from local testing to cloud-based production.
  
5. **Version Control and Model Lifecycle Management:**
- Managing different versions of models and promoting them through stages (e.g., Staging, Production) is essential in production ML systems.
- Decision: Introduce the Model Registry component with classes like RegisteredModel and ModelVersion to provide structured version control, model promotion, and lifecycle management.

6. **Ease of Integration and Open Standards:**
- MLflow was designed to integrate easily with existing tools, which meant it needed to be built on open, simple standards.
- Decision: Use familiar concepts (e.g., Experiment, Run, Artifact) and straightforward file-based configurations (e.g., MLproject and MLmodel YAML files).

### Main Classes and Components in MLflow

- **Tracking (Experiment, Run):**
  - Core purpose: Track and log experiments.
  - Why? Provides visibility and accountability in iterative ML work, which is essential for reproducibility and tuning.
- **Projects (MLproject, Environment):**
  - Core purpose: Enable reproducible and portable ML code execution.
  - Why? Standardized environments and entry points make it easier to share and rerun experiments across different setups.
- **Models (MLmodel, Flavors):**
  - Core purpose: Provide a unified format for packaging models across frameworks.
  - Why? Simplifies deployment and ensures that models are production-ready with minimal transformation.
- **Model Registry (RegisteredModel, ModelVersion):**
  - Core purpose: Manage model versions and stages.
  - Why? Allows teams to handle models in production environments systematically, including versioning, staging, and transitioning to production.

### Overall Package Structure

The MLflow package structure and its modular components (Tracking, Projects, Models, and Model Registry) were likely influenced by the following key considerations:

1. **User-centric Design:** MLflow’s structure reflects the journey of a data scientist or ML engineer, providing tools they need at each step, from tracking experiments to deploying models.
2. **Flexible Use Cases:** By making each component standalone (while still interoperable), MLflow allows users to adopt only what they need, whether it’s tracking, model packaging, or deployment.
3. **Open Source and Community-driven:** MLflow’s API and class design are open and extendable, allowing it to grow with new ML practices, integrations, and contributions from the community.
4. **Emphasis on Reproducibility:** Each component is geared toward making ML experiments reproducible, allowing users to save all artifacts and configurations required to rerun or refine past experiments.

## Summary

MLflow’s design addresses the core ML lifecycle needs:

- **Experiment** Tracking (reproducibility and comparison),
- Standardization of **Environments** (portability and consistency),
- **Model Packaging** (deployment readiness), and
- **Lifecycle Management** (version control and model stages).

## In case we are wondering the paths for mlflow

tracking_uri = mlflow.get_tracking_uri() # for the log_metric, log_param
artifact_uri = mlflow.get_artifact_uri() # for the log_artifact
run_id = mlflow.active_run().info.run_id # the run_id
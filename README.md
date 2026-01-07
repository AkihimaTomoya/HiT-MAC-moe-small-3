# Hierarchical Mixture of Vital Feature Experts

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the worker, run this command:

```train
python main.py --env Pose-v0 --model single-fm --workers 6
```

To train the orchestrator, run this command:

```train
python main.py --env Pose-v1 --model fm-att --workers 6
```


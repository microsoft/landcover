# Notes


To add a new model
- Create a new entry in models.mine.json that has three keys:
  - model
  - classes
  - metadata
- The "model" key must have a "fn" entry and a "type" entry
- The worker.py loader code must know how to map the "type" entry to a class that extends ModelSessionAbstract.ModelSession
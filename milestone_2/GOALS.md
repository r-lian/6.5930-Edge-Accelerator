**Optimize the hardware** (Ethos-U55) **for the workload** (YOLO-World)
- [ ] Done
    - In progress with `codesign.py`
1. Evaluate optimal resource allocation for YOLO WORLD ****for a fixed resource budget
    1. Evaluate 9 permutations of area, power, and throughput budgets (3 per param) effect on workload performance (mainly speed)
        1. Permit permutations outside of what ARM has specified for the Ethos-U55, to explore a vaster space than ARM has

**Optimize the workload for the hardware** 
- [ ] Done 
    - In progress with `codesign.py`
1. Modify the workload according to our most and least permissive hardware budgets
    1. Maximize speed (can’t track accuracy)

**Additional (from last milestone check-in)**
- [X] Done

1. Unify accelerator + workload into a compatible script
2. Find data-grounded energy and area defaults using `hwcomponents` or similar accelerators
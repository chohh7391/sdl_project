29-Aug-2026

Dear Prof. KIM:

I am writing to you regarding manuscript # Access-2026-33062 entitled "LLM-Guided Tool-Aware Task and Motion Planning for Chemistry Lab Automation" which you submitted to IEEE Access.

Your article was peer reviewed with interest but has not been recommended for publication in its current form.  We strongly encourage you to address the reviewers’ concerns, which can be found at the bottom of this letter, and resubmit your article to IEEE Access once you have updated it accordingly.
 
Please note that IEEE Access has a binary peer review process. Therefore, to uphold quality to IEEE standards, an article is rejected even if it requires minor edits.
 
When updating your manuscript, you should elaborate on your points and clarify with references, examples, data, etc. If you disagree with any technical points the reviewers have made, please include your counterarguments in your response to the reviewers (more information detailed below) and work this into the updated manuscript. 

Also, note that if a reviewer suggested references, you should only add those that are relevant to your work if you feel they strengthen your article. Recommending references to specific publications is not appropriate for reviewers and you should report excessive cases to ieeeaccessEIC@ieee.org.  Authors are not obligated to cite articles that are recommended by the reviewers, and the final decision on the article will not be influenced by whether or not authors cite these suggested references.
 
IEEE Access allows one opportunity to resubmit. If the updated manuscript is determined not to have addressed all of the previous reviewers’ concerns, or if the Associate Editor still has substantial technical concerns, the article will be rejected and no further resubmissions will be allowed.
 
When you are ready to resubmit your updated article, you can do so in the IEEE Author Portal.  When you log into the IEEE Author Portal you will see the title of the rejected article and the option to “Start Resubmission”.

https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-72202735-2BD4-4528-8EEB-5054133C8060-B6A52D9D-C72C-47ED-B635-427E0B06F60F-59844/current?idtype=external
 
Upon resubmission you will be asked to upload the following 3 files:

1) A document containing your response to reviewers from the previous peer review.  The “response to reviewers” document (template attached) should have the following regarding each comment: a) Reviewer’s concern, b) your response to the concern, c) your action to remedy the concern. The document should be uploaded with your manuscript files under "Author's Response Files.”

2) Your updated manuscript with all your individual changes highlighted, including grammatical changes (e.g. preferably with the yellow highlight tool within the pdf file). This file should be uploaded with your manuscript files as “Highlighted PDF.”

3) A clean copy of the final manuscript (without highlighted changes) submitted as a Word or LaTeX file, and as a PDF, both submitted as the “Main Manuscript.”

**IMPORTANT: Please see the attached Resubmission Checklist that details all the items listed above.  Please utilize this checklist to ensure you have made the necessary edits to your manuscript, and to ensure you have all the necessary files prepared prior to resubmission.

*** AUTHOR LIST CHANGES: If your revised manuscript has an updated author list, you will need to submit a formal request to the Editor by completing the attachment labelled ‘Request for Byline Change,’ and uploading it as 'Request for byline change form.' This should include a DETAILED justification explaining each author’s contribution(s) to the work. You will also need to provide the justification for the author change during the submission process.  Change in the author list is considered rare and exceptional, and the decision to allow such changes rests with the Editor. Once the list and order of authors has been established, the list and order of authors should not be altered without permission of all living authors of that article.

We sincerely hope you will update your manuscript and resubmit soon. Please contact me if you have any questions.

Thank you for your interest in IEEE Access.

Sincerely,

Dr. M. Anwar Hossain
Associate Editor, IEEE Access
ahossain@queensu.ca

Reviewers' Comments to Author:

Reviewer: 1

Comments:
The manuscript presents a coherent hierarchical framework combining natural-language protocol generation, context-aware tool and rearrangement reasoning, constrained task and motion planning, perception, and sensor-feedback execution skills. The proposed integration is relevant to flexible laboratory automation. However, several aspects of the current experimental design and interpretation need to be strengthened before the reported results can support the broader claims made for the complete framework.

1. End-to-end validation of the Perception Module: Section III-C describes a multi-camera AprilTag-based perception pipeline with pose fusion and world-state construction, but Section IV-A states that the simulation World State is synchronized by directly extracting object poses and robot joint configurations from NVIDIA Isaac Sim. Consequently, the quantitative end-to-end experiments do not appear to exercise the proposed camera-based perception chain. The authors should clearly distinguish ground-truth simulator state from perception-derived state and either evaluate the complete perception pipeline, including localization error, occlusion robustness, and its influence on planning success, or explicitly restrict the end-to-end claims to planning and execution under externally supplied state estimates. This is a major but correctable issue because the Perception Module is presented as one of the four necessary components of the framework.

2. Scope of execution-layer validation: Sections III-E, IV-E, and Table 8 treat adaptive pouring, tool changing, and device operation as part of the integrated execution layer, while Section V-A acknowledges that all quantitative evaluation was performed in simulation and that the simulator does not reproduce the fluid-dynamic and sensing effects governing real liquid transfer. The statement that the Skill Library introduces no additional failure modes and that planning-stage verification is sufficient to predict end-to-end outcomes is therefore stronger than the evidence supports. The authors should define precisely which execution uncertainties are represented in the simulation, report the criteria used to classify execution success, and restrict conclusions about execution robustness to the phenomena actually modeled. Validation with the physical FR5 platform would be required to support claims concerning quantitative pouring accuracy or robustness to real sensor and fluid effects. This is a major but correctable issue.

3. Fairness and reproducibility of the planner comparison: Section IV-C and Table 4 compare cuTAMP with PDDLStream, but the manuscript does not provide sufficient implementation detail to establish that the comparison uses comparable computational resources and appropriately configured planners. This is particularly important because cuTAMP evaluates 1,024 particles in parallel, whereas the computational configuration, stream definitions, sampling settings, retry policy, planner version, hardware allocation, and CPU or GPU resources used for PDDLStream are not reported with equivalent detail. The authors should provide enough information to reproduce both configurations and explain how the baseline parameters were selected. This is a major issue because the central claim of improved planning reliability depends directly on this comparison.

4. Statistical treatment of planning success and latency: Table 4 reports only 30 trials per condition and calculates planning-time mean and standard deviation over successful trials only, while failed PDDLStream runs include both timeouts and early terminations. Excluding these trials from the latency statistics makes direct interpretation of the reported planning-time distributions difficult and can bias conclusions regarding temporal predictability. The authors should report uncertainty for the success proportions, provide an appropriate statistical comparison between planners, and present a time-to-solution analysis that accounts explicitly for failed and timed-out trials rather than characterizing only successful runs. The claim of reduced planning-time variance should also be formulated consistently with this treatment. This is a major but correctable issue.

5. XDL Generator evaluation and test-set independence: Section IV-B1 reports 97 valid protocols from 100 test instructions, but the three unsuccessful cases are attributed to contradictions already present in the user instructions and are rejected at the physical-feasibility validation stage. This evaluation therefore mixes protocol-generation accuracy with the downstream validator's ability to reject an invalid requested sequence. The authors should separate generator accuracy from validator performance and report whether the generated operators, arguments, object identities, numerical values, and procedural ordering match independently defined ground truth for valid commands. The manuscript should also explain how training, validation, and test instructions were separated at the template, synonym, object, and compositional levels to demonstrate that the reported performance is not due to overlap between combinatorially generated training and test patterns. This is a major but correctable issue.

6. Overstatement of XDL Validator coverage: Section IV-B2 concludes that the three-layer validator provides complete and non-overlapping coverage of physically relevant failure modes based on a benchmark containing 20 valid and 80 invalid samples constructed from a limited set of injected error categories. The tested conditions support successful detection of those specified error classes, but they do not establish complete physical-feasibility coverage for laboratory procedures. Capacity limits, chemical incompatibility, unavailable tools, unreachable configurations, device-state conflicts, contamination constraints, and other physically relevant conditions are outside the demonstrated benchmark. The authors should either broaden the validation substantially or narrow the claim to the explicitly tested syntactic, object-existence, and symbolic state-precondition failures. This is a major but readily correctable issue.

7. Generalization and context-aware rearrangement claims: The Action Reasoner is evaluated within the same six-operator and 12-grid abstraction used to construct its combinatorial training dataset, and the rearrangement ablation in Table 7 is limited to two short procedure sequences with 30 trials per condition. These experiments demonstrate useful performance within the defined task distribution, but they provide limited evidence for broader context-aware generalization or for the conclusion that procedural context is the decisive factor across multi-step chemical workflows. The authors should specify exactly what is unseen in the Action Reasoner test set, including object classes, spatial configurations, and task combinations, and either broaden the sequence-level evaluation or restrict the generalization claims to the tested workspace and operator distribution. This is a moderate to major but correctable issue.

8. Reproducibility of the experimental methodology: Several implementation details required to independently reproduce the reported results are either omitted or deferred to the project page. The manuscript should provide the essential experimental parameters directly, including the Isaac Sim version and simulation settings, workspace dimensions and randomization ranges, collision margins, robot and controller settings, planner objective and constraint parameters, stopping criteria, retry handling, random-seed policy, and precise definitions of planning and execution success. The same level of detail should be given for both the proposed method and the baseline. This is a major but correctable issue because the current description is sufficient to understand the architecture but not yet sufficient to reproduce the quantitative results.

Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: Yes

2) Is the paper technically sound?: No

3) Is the subject matter presented in a comprehensive manner?: No

4) Are the references provided applicable and sufficient?: Yes

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.: Not applicable.


Reviewer: 2

Comments:
1. The academic direction of integrating LLMs and TAMP for flexible chemistry lab automation is highly encouraging and valuable.

2. However, as the current validation is limited to simulations, additional experimental results in a physical robot environment are essential to address the Sim-to-Real gap.

3. The manuscript requires significant improvements in administrative, specifically regarding the missing Author Response Table.

4. It is strongly recommended to elaborate on the defense mechanisms against LLM hallucinations to ensure system reliability and to perform a major revision addressing these critical points.

Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: Yes, this paper contributes to the field by seamlessly integrating Large Language Models (LLMs) with Task and Motion Planning (TAMP) for chemistry lab automation. It effectively overcomes the limitations of manual pipeline design by translating natural language into standard XDL protocols and linking them directly to the robot's physical manipulation and obstacle avoidance plans.

2) Is the paper technically sound?: Yes, the overall algorithmic and system design integrating LLMs with the TAMP module via XDL is technically sound. However, the validation is restricted to simulation (NVIDIA Isaac Sim), meaning it lacks physical robot experiments needed to verify real-world variables like fluid viscosity and sensor latency.

3) Is the subject matter presented in a comprehensive manner?: The theoretical background and system architecture (Perception, LLM, and TAMP modules) are described comprehensively and clearly. However, the presentation falls short due to the missing IEEE Access mandatory author response table and a lack of depth regarding error-handling mechanisms for LLM hallucinations.

4) Are the references provided applicable and sufficient?: The references are applicable and sufficient, covering recent advancements in LLMs, robotics, and computer vision.

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:


Reviewer: 3

Comments:
Summary of the Work
The manuscript introduces an integrated hierarchical decision-making and planning framework for autonomous chemistry laboratories. By coupling a fine-tuned lightweight LLM (Llama 3.2 1B) for Chemical Description Language (XDL) protocol synthesis and context-aware action reasoning (end-effector selection and target grid placement) with cuTAMP for differentiable motion planning, the framework achieves robust constraint handling and bounded planning times. Validation across randomized simulation trials in NVIDIA Isaac Sim demonstrates high success rates (97.78% end-to-end) and significant improvements over parameter-binding baselines like PDDLStream under tight orientation constraints.

Key Strengths
* Effective Hierarchical Integration: The separation of discrete symbolic decision-making (LLM) from continuous geometric optimization (cuTAMP) and sensor-feedback primitives cleanly bounds model responsibilities.
* Technical Rigor: Clear mathematical formulations for the SidePick approach angle sampling, MoveHolding upright orientation cost J_up(q), and inverse-distance weighted dual-camera pose fusion.
* Bounded Planning Latency: Utilizing cuTAMP successfully mitigates the stochastic, unbounded termination times characteristic of sequential sampling planners on low-redundancy 6-DoF manipulators.
* Comprehensive Evaluation: Extensive ablations covering protocol generation accuracy, solver reliability, tool selection, rearrangement strategy, and multi-step workflow execution.

Minor Revisions and Suggestions
1. Fluid Dynamics & Sim-to-Real Transfer: The manuscript notes that Isaac Sim abstracts fluid-dynamic effects. Please expand briefly on how real-world fluid properties (viscosity variations, sloshing, surface tension, scale latency) might impact the Adaptive Pouring feedback loop during physical deployment on the FR5 arm.
2. Upright Constraint Threshold: The maximum allowable tilt angle is fixed at theta_max = 5°. Clarify whether this threshold was selected based on a specific vessel fill-level ratio or fluid viscosity range.
3. Perception Resilience: The state grounding relies on AprilTags. Discuss potential fallback or recovery behavior if tags become obscured by steam, condensation, or chemical residue in physical laboratory conditions.
4. Minor Presentation Refinements:
- Double-check that all table captions and column headers explicitly denote measurement units (e.g., planning time in seconds in Table 4).
- Ensure consistency in notation across figure diagrams (e.g., Figure 1 and Figure 3 process blocks).

Additional Questions:
Please confirm that you have reviewed all relevant files, including supplementary files and any author response files, which can be found in the "View Author's Response" link above (author responses will only appear for resubmissions): Yes, all files have been reviewed

1) Does the paper contribute to the body of knowledge?: yes

2) Is the paper technically sound?: yes

3) Is the subject matter presented in a comprehensive manner?: yes

4) Are the references provided applicable and sufficient?: [1] Peng, Qucheng, et al. "NavigScene: Bridging local perception and global navigation for beyond-visual-range autonomous driving." Proceedings of the 33rd ACM International Conference on Multimedia. 2025.

[2] Peng, Q., Xue, H., Wang, P., & Chen, C. (2026, March). Lifelong Domain Adaptive 3D Human Pose Estimation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 10, pp. 8358-8366).

5) Are there references that are not appropriate for the topic being discussed?: No

5a) If yes, then please indicate which references should be removed.:

If you have any questions, please contact article administrator: Ms. Suhasini Das das.suhasini@ieee.org
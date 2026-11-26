# Computer Networks submission: upload steps

Submission link (from the journal's ScienceDirect page, "Submit your article"): https://submit.elsevier.com/COMNET
Guide for Authors: https://www.sciencedirect.com/journal/computer-networks/publish/guide-for-authors

## Files in this folder

| File | Upload as |
|---|---|
| manuscript.pdf | Manuscript (PDF for reviewers, 19 pages, two-column) |
| submission_source.zip (manuscript.tex, manuscript.bbl, refs.bib, figures/) | LaTeX source files (compiles with pdflatex alone; the .bbl is included) |
| highlights.docx | Highlights (separate editable file, "highlights" in the name; 5 bullets, each at most 85 characters) |
| credit_author_statement.docx | CRediT author statement |
| declarations.docx | Competing interest, funding, data availability, generative-AI statements (also in the manuscript), for pasting into the forms |
| abstract.txt | Plain-text abstract for the abstract field |

## Steps

1. Open https://submit.elsevier.com/COMNET and sign in or register with your Elsevier account.
2. Pick the regular research article type.
3. Upload the files from the table above. If the system asks for one "manuscript" file, upload manuscript.pdf as the manuscript and the zip, or its files one by one, as LaTeX source.
4. Paste the metadata:
   - Title: Speak When Sure, Ask When Unsure: Confidence-Aware Status Updates from On-Board Classifiers in Aerial IoT
   - Author: Sameh Hosny (corresponding), Electrical and Computer Engineering Department, American University of Armenia, Yerevan, Armenia, sameh.hosny@aua.am
   - Keywords (7): Age of incorrect information; on-board inference; confidence calibration; unmanned aerial vehicle; Internet of Things; Lyapunov optimization; POMDP
   - Abstract (249 words): see below
5. Complete the declarations tool. Competing interests: none. Generative AI use: yes, with the statement in declarations.docx.
6. Funding: no specific grant.
7. Publication route: subscription (no article publishing charge). Open access would carry an APC.
8. Optional: cover letter, suggested reviewers, graphical abstract.
9. The journal asks for a biography of at most 100 words per author and a passport-type photo, as separate editable files. These are not in this package yet.
10. Check the PDF the system builds, then submit.

## Abstract (plain text)

Internet of Things (IoT) devices classify data on board and send a few-byte label instead of raw data, but a small classifier errs, and wrong labels mislead the receiver. An unmanned aerial vehicle (UAV) monitors a scene for a ground station (GS) and, per slot, stays silent, sends its label, or uploads the frame to a more accurate ground classifier, to minimize the average age of incorrect information (AoII) at the GS under an energy budget and an intermittent link. The UAV never observes the true state, so we formulate a constrained partially observable Markov decision process and show that a belief over the current AoII, updated with each frame's calibrated confidence, is a sufficient statistic. We derive each action's one-step expected AoII in closed form, prove that a label agreeing with the GS is never worth sending, and show that drift-plus-penalty reduces to confidence thresholds ordered silent, raw, label. The policy, CA-DPP, meets the energy budget on every sample path, and comes within 25% of a lower bound on the optimum when its assumptions hold. On Sentinel-2 flood-mapping traces, it lowers the AoII by 52% to 75% relative to pre-identification and age-threshold policies at budgets up to 2 mJ per slot. Against an AoII-threshold policy, it lowers the AoII by 22% ± 12% at 0.25 mJ per slot and up to 49% when the on-board classifier is weak. When the classifier is accurate and energy is ample, CA-DPP and the AoII-threshold policy perform the same within statistical error.

# IEEE Transactions on Mobile Computing: upload steps

Checked on 2026-11-26 against the journal's own pages:
- Author information: https://www.computer.org/csdl/journal/tm/write-for-us/15075
- Overlength policy: https://www.computer.org/csdl/journal/tm/write-for-us/201961
- Submission link (the "Submit a Manuscript" link on the TMC page): https://ieee.atyponrex.com/journal/tmc-cs
- IEEE AI policy: https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/

## What the rules say, and how this version meets them

| Rule | This version |
|---|---|
| Regular paper: 12 double-column pages free, including references and author biographies; each extra page or fraction costs $220 (submissions up to 18 pages) | 12 full pages in IEEEtran compsoc journal format (the official IEEE Computer Society LaTeX template), with the biography and a 1 in x 1.25 in photo |
| Supplemental material in separate files, not in the main PDF; no page limit; appendices count as supplemental | tmc_supp.pdf (4 pages): proofs, notation table, extra figures and robustness checks |
| Abstract 100 to 200 words, no math or citations | 199 words, plain text, no math |
| Keywords from the ACM taxonomy offered in the submission system, at least three | Pick these terms or the closest ones in the list: age of information / age of incorrect information; on-board inference; confidence calibration; unmanned aerial vehicles; Internet of Things; Lyapunov optimization; partially observable Markov decision processes |
| AI-generated content must be disclosed in the acknowledgments section, naming the system and the sections | In the Acknowledgments of tmc.tex |
| Biographies and photos are required only at final acceptance | Already included, and counted in the 12 pages |
| Review is single-anonymous by default | Author name stays on the manuscript |

## Files

| File | Upload as |
|---|---|
| manuscript.pdf | Main document (PDF for review, 12 pages) |
| supplementary.pdf | Supplemental material (separate file) |
| source.zip (tmc.tex, tmc.bbl, refs.bib, tmc_supp.tex, figures/, photo) | LaTeX source, if the system asks for it (it is needed at the final stage) |
| abstract.txt | Paste into the abstract field |

## Steps

1. Open https://ieee.atyponrex.com/journal/tmc-cs and sign in with your IEEE account (create one if needed).
2. Choose the manuscript type "Regular paper".
3. Paste the title: Speak When Sure, Ask When Unsure: Confidence-Aware Status Updates from On-Board Classifiers in Aerial IoT
4. Paste the abstract from abstract.txt and choose the keywords listed above.
5. Author: Sameh Hosny (corresponding), Electrical and Computer Engineering Department, American University of Armenia, Yerevan, Armenia, sameh.hosny@aua.am.
6. Upload manuscript.pdf as the main document and supplementary.pdf as supplemental material.
7. Answer the forms: no funding; no conflicts of interest; no human or animal subjects; the paper is not under review elsewhere.
8. If the form asks about AI use, repeat the statement in the Acknowledgments.
9. Choose the traditional (non open access) route; open access would carry an article processing charge.

The paper must not be under review at Computer Networks or any other venue at the same time. Submit to only one of the two.

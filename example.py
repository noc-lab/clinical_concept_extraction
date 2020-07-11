import warnings
warnings.filterwarnings("ignore")

from spacy import displacy
# from IPython.core.display import display, HTML

# building (elmo embbedings prediction graph ) and clinical conceot extraction graph
# might take some while about 20~30 seconds
from clinical_concept_extraction import ClinicalConceptExtraction
from clinical_concept_extraction.utils import build_display_elements


clinical_concept_extraction = ClinicalConceptExtraction(models_path='/home/omar/Desktop/clinical_concept_extraction/cce_assets')


# An example of a discharge summary contains arbitrary line breaks. I faked this reports.
sample_text = """
This is an 119 year old woman with a history of diabetes who has a CT-scan at 2020-20-20. Insulin is prescribed for the type-2 diabetes. Within the past year, the diabetic symptoms have progressively gotten worse.
"""

all_annotations_of_sample_text = clinical_concept_extraction.extract_concepts(sample_text, batch_size=3, as_one_batch=False)
print(all_annotations_of_sample_text)

all_annotations_of_sample_text = clinical_concept_extraction.extract_concepts(sample_text, batch_size=3, as_one_batch=False)
print(all_annotations_of_sample_text)

all_annotations_of_sample_text = clinical_concept_extraction.extract_concepts(sample_text, batch_size=3, as_one_batch=False)


print(all_annotations_of_sample_text)
import re
from nif import NIFContent
def lm_output_to_annoation_full(string_with_annotations:str):
    pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\] \[ ([^\]]+) \]"
    annotations=[]

    result = re.search(pattern, string_with_annotations)
    while result is not None:
        annotation={"begin":+result.start(),"end":result.start()+len(result.group(1)),"mention":result.group(1),"link":result.group(2)}
        annotations.append(annotation)
        string_with_annotations=string_with_annotations.replace(result.group(0),result.group(1),1)
        result = re.search(pattern, string_with_annotations)
    print(string_with_annotations)
    return annotations
def lm_output_to_annoation_ner(string_with_annotations:str,ref_context:str):
    pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\]"
    annotations=[]
    base_uri = ref_context[0:ref_context.find('#')]
    result = re.search(pattern, string_with_annotations)
    while result is not None:
        begin_index=result.start()
        end_index=result.start()+len(result.group(1))
        mention=result.group(1)
        content=NIFContent.NIFContent(uri=base_uri+"#"+str(begin_index)+","+str(end_index))
        content.set_begin_index(begin_index)
        content.set_end_index(end_index)
        content.set_reference_context(ref_context)
        content.set_anchor_of(mention)
        annotations.append(content)
        string_with_annotations=string_with_annotations.replace(result.group(0),result.group(1),1)
        result = re.search(pattern, string_with_annotations)
    print(string_with_annotations)
    return annotations

def lm_output_to_annoation_e2e(string_with_annotations:str,ref_context:str,wikidata_uris):
    pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\] \[ ([^\]]+) \]"
    annotations=[]
    base_uri = ref_context[0:ref_context.find('#')]
    result = re.search(pattern, string_with_annotations)
    while result is not None:
        begin_index=result.start()
        end_index=result.start()+len(result.group(1))
        mention=result.group(1)
        content=NIFContent.NIFContent(uri=base_uri+"#"+str(begin_index)+","+str(end_index))

        content.set_begin_index(begin_index)
        content.set_end_index(end_index)
        content.set_reference_context(ref_context)

        #if result.group(2)in titles_to_wikipedia:
        #    wikidata_link=titles_to_wikipedia[result.group(2)]
        #else:
        #    wikidata_link="0000"
        content.set_taIdentRef("http://en.wikipedia.org/wiki/"+result.group(2).replace(" ","_"))

        content.set_anchor_of(mention)
        if content.taIdentRef in wikidata_uris:
            annotations.append(content)
        #annotations.append(content)
        string_with_annotations=string_with_annotations.replace(result.group(0),result.group(1),1)
        result = re.search(pattern, string_with_annotations)
    print(string_with_annotations)
    return annotations

#test2="aldka"
test_str="[START_ENT] Paderborn [END_ENT] [ Paderborn ] was founded as a bishopric by [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] in 795, although its official history began in 777 when [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] built a castle near the [START_ENT] Paderborn [END_ENT] [ Paderborn ] springs.[4] In 799 [START_ENT] Pope Leo III [END_ENT] [ Pope Leo III ] fled his enemies in [START_ENT] Rome [END_ENT] [ Rome ] and reached [START_ENT] Paderborn [END_ENT] [ Paderborn ], where he met [START_ENT] Charlemagne [END_ENT] [ Charlemagne ], and stayed there for three months. It was during this time that it was decided that [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] would be crowned emperor. [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] reinstated [START_ENT] Leo [END_ENT] [ Pope Leo I ] in [START_ENT] Rome [END_ENT] [ Rome ] in 800 and was crowned as [START_ENT] Holy Roman Emperor [END_ENT] [ Holy Roman Emperor ] by [START_ENT] Leo [END_ENT] [ Pope Leo I ] in return."
print(lm_output_to_annoation_full(test_str))
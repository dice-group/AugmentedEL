import pickle
import re
# from genre.fairseq_model import GENRE
from GENRE.genre.hf_model import GENRE as genre_hf
from experiments.prefix_trie.PrefixTrie import Trie_not_recursive, Trie
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
from nif import NIFDocument, NIFContent
import pickle
import requests
import json

class GenreDisamb():
    def __init__(self):
        with open("../GENRE/kilt_titles_trie_dict.pkl", "rb") as f:
            self.trie = Trie.load_from_dict(pickle.load(f))

        # load the model
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            "t5-base", )
        self.model = genre_hf.from_pretrained("../GENRE/hf_entity_disambiguation_aidayago").eval()
        # self.model.to("cuda:0")

    def gentries(self, mention_candidate_dict):
        trie = Trie_not_recursive()
        # entities=list(entities.keys())
        # entities.append("relations: ")
        entdict = {}
        for mention in mention_candidate_dict:
            key_encoding = self.target_tokenizer.encode("[START_ENT] " + mention + " [END_ENT] [")[:-1]
            trie = Trie_not_recursive()
            for cand in mention_candidate_dict[mention]:
                lab = " " + cand + " ]"
                print(lab)
                seq = self.target_tokenizer.encode(lab)[:-1]
                trie.add(seq)
            entdict[",".join(str(el) for el in key_encoding)] = trie
        return entdict

    def predict_for_annotaions(self, strings_with_annotations):
        pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\]"
        annotations = []
        # base_uri = ref_context[0:ref_context.find('#')]
        annotation_to_candidates = {}
        for str_el in strings_with_annotations:
            result = re.search(pattern, str_el)
            annotations = []
            while result is not None:
                begin_index = result.start()
                len_annotation = len(result.group(1))
                annotations.append((begin_index, len_annotation, result.group(1)))
                str_el = str_el.replace(result.group(0), result.group(1), 1)
                result = re.search(pattern, str_el)
            search_strs = []
            for el in annotations:
                search_strs.append(str_el[0:el[0]] + "[START_ENT]" + str_el[el[0]:el[0] + el[1]] + "[ENT_ENT]" + str_el[
                                                                                                                 el[0] +
                                                                                                                 el[
                                                                                                                     1]:])
            print(search_strs)
            if len(search_strs) != 0:
                links = self.model.sample(
                    sentences=search_strs,
                    prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                    num_return_sequences=10, num_beams=10
                )
            else:
                links = []
            for i in range(len(annotations)):
                candididates = links[i]
                cand_strings = set([el["text"] for el in candididates])
                if not annotations[i][2] in annotation_to_candidates:
                    annotation_to_candidates[annotations[i][2]] = set()
                annotation_to_candidates[annotations[i][2]].update(cand_strings)
        trie_dict = self.gentries(annotation_to_candidates)
        return trie_dict

    def test_pred(self, search_strs):
        if len(search_strs) != 0:
            links = self.model.sample(
                sentences=search_strs,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()), num_return_sequences=10,
                num_beams=10
            )
        else:
            links = []
        return links

    def load(self, path_to_ds):
        with open(path_to_ds, 'r', encoding='utf-8') as file:
            doc = NIFDocument.nifStringToNifDocument(file.read())
        return doc

    def groupNifDocumentByRefContext(self, ds: NIFDocument):
        refContextToDocument = {}
        for nifContent in ds.nifContent:
            if nifContent.reference_context is not None:
                docrefcontext = nifContent.reference_context
            else:
                docrefcontext = nifContent.uri
            if docrefcontext in refContextToDocument:
                refContextToDocument.get(docrefcontext).addContent(nifContent)
            else:
                doc = NIFDocument.NIFDocument()
                doc.addContent(nifContent)
                refContextToDocument[docrefcontext] = doc
        return refContextToDocument

    def test_nif_doc(self):
        dataset_names = []
        # dataset_names.append("ACE2004")
        # dataset_names.append("aida_testa")
        # dataset_names.append("aida_testb")
        # dataset_names.append("aida_train")
        # dataset_names.append("aida_complete")
        # dataset_names.append("AQUAINT")
        # dataset_names.append("spotlight")
        dataset_names.append("der")
        # dataset_names.append("iitb-fix")
        dataset_names.append("KORE50")
        # dataset_names.append("MSNBC")
        dataset_names.append("N3-Reuters-128")
        dataset_names.append("N3-RSS-500")
        dataset_names.append("oke-challenge-task1-eval")
        # dataset_names.append("oke-challenge-task1-example")
        # dataset_names.append("oke-challenge-task1-gs")
        titles_to_wikipedia = pickle.load(open("../titles_to_wikipedia.pkl", "rb"))
        id_to_title = {v: k for k, v in titles_to_wikipedia.items()}

        for el in dataset_names:
            ds = self.load("../data/nif_wikipedia_cleaned/" + el)
            ents = self.groupNifDocumentByRefContext(ds)
            num_found = 0
            not_found = 0
            for doc in tqdm(list(ents.values())):
                doc_str = ""
                for cont in doc.nifContent:
                    if cont.is_string is not None:
                        doc_str = cont.is_string
                strs = []
                labels = []
                mentions=[]
                for cont in doc.nifContent:
                    if cont.is_string is None:
                        str_to_search = doc_str[
                                        max(0, int(cont.begin_index) - 100):int(int(cont.begin_index))] + "[START_ENT]" \
                                        + doc_str[int(cont.begin_index):int(cont.end_index)] + "[END_ENT]" + doc_str[
                                                                                                             int(cont.end_index):min(
                                                                                                                 int(cont.end_index) + 100,
                                                                                                                 len(doc_str[
                                                                                                                     int(cont.end_index):]))]
                        if cont.taIdentRef is not None:
                            if cont.taIdentRef.replace("http://", "") in id_to_title:
                                labels.append(id_to_title[cont.taIdentRef.replace("http://", "")])
                            else:
                                labels.append("emp")
                            mentions.append(cont.anchor_of)
                        strs.append(str_to_search)
                print(doc_str)
                found = self.test_pred(strs)
                entities=[]
                for el in found:
                    entities.append(", ".join(["\""+e["text"]+"\""for e in el]))
                for i in range(len(mentions)):
                    data={"sentence":"\""+doc_str+"\"","span":"\""+mentions[i]+"\"","entities":entities[i]}
                    ans = requests.post("http://localhost:6000/annotate/", data=json.dumps(data))
                    try:
                        print(ans.json())
                    except:
                        print(ans.content)
                # print(labels)
                # print(found)
                print("end")
            print(el + " num_found:" + str(num_found) + " ,num_not_found:" + str(not_found))
            # ds.nifContent=cont_rem
            with open("../data/wikipedia_strings/" + el, "w", encoding="utf-8") as out_file:
                out_file.write(ds.get_nif_string())


disamb = GenreDisamb()
disamb.test_nif_doc()

'''
test_str=["[START_ENT] Paderborn [END_ENT] was founded as a bishopric by [START_ENT] Charlemagne [END_ENT] in 795, although its official history began in 777 when [START_ENT] Charlemagne [END_ENT] built a castle near the [START_ENT] Paderborn [END_ENT] springs.(4) In 799 [START_ENT] Pope Leo III [END_ENT] fled his enemies in Rome and reached [START_ENT] Paderborn [END_ENT], where he met [START_ENT] Charlemagne [END_ENT], and stayed there for three months. It was during this time that it was decided that [START_ENT] Charlemagne [END_ENT] would be crowned emperor. [START_ENT] Charlemagne [END_ENT] reinstated [START_ENT] Leo [END_ENT] in [START_ENT] Rome [END_ENT] in 800 and was crowned as [START_ENT] Holy Roman Emperor [END_ENT] by [START_ENT] Leo [END_ENT] in return."]
disamb=GenreDisamb()

links,trie=disamb.predict_for_annotaions(test_str)
print(links)



for el in link[0]:
    el["wikidata_id"]=title_to_wikidata_id[el["text"]]
print(link)
'''
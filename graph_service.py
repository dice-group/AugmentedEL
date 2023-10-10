from flask import Flask,request,jsonify
import pickle
import traceback
import json
app = Flask(__name__)
graph = pickle.load(open("dbpedia_graph.pkl","rb"))
nodes=pickle.load(open("node_set_only_graph.pkl","rb"))
@app.route('/neighbours', methods=['GET', 'POST'])
def neighbours():
    try:
       input = request.data.decode()
       out = graph.neighbors(input, mode='all')
       neighbours=[]
       print(out)
       for n in out:
           neighbours.append(graph.vs[n]["name"])
       print(neighbours)
       return jsonify(neighbours)
    except Exception as e:
        return traceback.format_exc(), 500
@app.route('/jaccard', methods=['GET', 'POST'])
def jaccard():
    try:
       input = json.loads(request.data.decode())
       out = graph.similarity_jaccard(input,loops=False)
       return jsonify(out)
    except Exception as e:
        return traceback.format_exc(), 500

@app.route('/get_subgraph', methods=['GET', 'POST'])
def subgraph():
    try:
        input = json.loads(request.data.decode())
        filtered=[]
        for n in input:
            if n in nodes:
                filtered.append(n)
        res = graph.induced_subgraph(
            filtered)
        triples=[]
        for edge in res.es:
            triple=[]
            source_vertex_id = edge.source
            target_vertex_id = edge.target
            triple.append(res.vs[source_vertex_id]["name"])
            triple.append(edge["predicate"])
            triple.append(res.vs[target_vertex_id]["name"])
            if not triple[0] == triple[2]:
                triples.append(triple)

        return jsonify(triples)
    except Exception as e:
        return traceback.format_exc(), 500

@app.route('/get_subgraph_dp_1', methods=['GET', 'POST'])
def subgraph_d1():
    try:
        input = json.loads(request.data.decode())
        filtered=set()
        for n in input:
            if n in nodes:
                filtered.add(n)
        input=filtered.copy()
        neighbbourhood=graph.neighborhood(vertices=filtered, order=1, mode='all', mindist=1)
        hit_nodes= {}
        for el in neighbbourhood:
            for n in el:
                if not n in hit_nodes:
                    hit_nodes[n]=0
                hit_nodes[n]+=1
        for el in hit_nodes:
            if hit_nodes[el]>1:
                filtered.add(graph.vs[el]["name"])
        res = graph.induced_subgraph(
            list(filtered))
        direct_triples=[]
        i = set(input)
        edges_to_delete=[]
        for edge in res.es:
            triple=[]
            source_vertex_id = edge.source
            target_vertex_id = edge.target
            triple.append(res.vs[source_vertex_id]["name"])
            triple.append(edge["predicate"])
            triple.append(res.vs[target_vertex_id]["name"])
            #triples.append(triple)
            if not triple[0] == triple[2] and triple[0] in i and triple[2] in i:
                direct_triples.append(triple)
                edges_to_delete.append(edge)
            if triple[0] == triple[2]:
                edges_to_delete.append(edge)
        res.delete_edges(edges_to_delete)
        paths=[]
        for source in input:
            for dest in input:
                if source !=dest:
                    p_found=res.get_shortest_paths(source, to=dest, weights=None, mode='out', output='epath')
                    for found in p_found:
                        s = source
                        d = dest
                        if len(found) ==2:
                            paths.append([s , d])
                        '''
                        edge_string = ""
                        for edge in found:
                            edge_string+=res.es[edge]["predicate"]+" "
                        if edge_string !="":
                            paths.append([s,edge_string,d])
                        '''
        output={"direct_relations":direct_triples,"one_hop_connected":paths}
        return jsonify(output)

    except Exception as e:
        return traceback.format_exc(), 500

if __name__ == "__main__":
    app.run()
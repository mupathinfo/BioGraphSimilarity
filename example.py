#!/usr/bin/python
#
# 2021 Mikhail Kovalenko kovalenkom@health.missouri.edu
# A primer on the use of the BioGraphSimilarity class functions
#

import time
# TODO switch away from py2neo to something better
# https://community.neo4j.com/t/are-there-any-real-instructions-for-using-py2neo/2689
from py2neo import Graph, Schema
# import numpy as np
# from pandas import DataFrame
import pandas as pd

from modules.EntityLookUpService import EntityLookUp
from modules.BioGraphSimilarity import GraphSimilarity


def main():
    # Connect to neo4j
    test_graph = Graph(address="localhost:17687", auth=("neo4j", "test"))
    kegg_graph = Graph(address="localhost:57686", auth=("neo4j", "test"))

    # initialize gene name-checking service
    gc = EntityLookUp()

    # load GraphSimilarity class
    # scoring: overlap, inclusion, iou (default)
    sim = GraphSimilarity()

    # the list of sources to extract from tetgraph
    papers = [
        "paper_name_1",
        "paper_name_2",
        "paper_name_3"
    ]

    print("Loading KEGG graph into RAM...")
    start_time = time.time()
    graph_reference = extract_graph(kegg_graph)
    print("KEGG loaded")
    print(">>> time: ", int(time.time() - start_time), "seconds")

    for p in papers:
        # extract the graphs for comparing
        # using the "source" relationship property
        print("\n=============================================================")
        print(str(p) + "\n")

        graph2compare = query_neo4j(test_graph, rel_properties={
            'source': p,
        })

        nodes1 = sim.extract_nodes(graph_reference)
        nodes2 = sim.extract_nodes(graph2compare)
        count1 = len(nodes1)
        count2 = len(nodes2)

        # calculate scores
        # print("\nNodes")
        score_n = sim.nodes(graph_reference, graph2compare)

        # print("\nUndirected relationships")
        score_ru = sim.relationships(graph_reference, graph2compare, directed=False)
        # print("\nDirected relationships")
        score_rd = sim.relationships(graph_reference, graph2compare, directed=True)

        # print("\nUndirected neighbors")
        score_nu = sim.neighbors(graph_reference, graph2compare, directed=False)
        # print("\nDirected neighbors")
        score_nd = sim.neighbors(graph_reference, graph2compare, directed=True)

        scores = [[score_n, score_ru, score_nu], [None, score_rd, score_nd]]
        df = pd.DataFrame(scores, ["Undirected", "Directed"], ["Nodes", "Relations", "Neighbors"])
        print(df)

        print("\nTwin search (top 5)")
        list_d = sim.find_twins(graph_reference, graph2compare, directed=True, head=5)
        for k, v in list_d.items():
            print(f"{k}:\t{v}")

        print("\nMatrix view\nv KEGG \\\ttest >")
        matrix = sim.find_twins(graph_reference, graph2compare, directed=True, matrix=True)
        print(matrix)

        print(f"\nSource\tKEGG_nodes\ttest_nodes\tNodes\tUnRel\tUnNeighbors\tDirRel\tDirNeighbors")
        row = str(p) + "\t" + str(count1) + "\t" + str(count2) + "\t" + str(score_n) + "\t" + str(score_ru)\
            + "\t" + str(score_nu) + "\t" + str(score_rd) + "\t" + str(score_nd)
        print(f"{row}")

    return 0


def query_neo4j(graph, startnode=None, endnode=None, rel_properties=None):
    """
    Returns a list[] of dicts{} for each relationship

    :param graph: Neo4j connection object
    :param startnode: string Start node name
    :param endnode: string End node name
    :param rel_properties: set Relationship properties
    :return: Neo4j recordset object
    """

    # build the query using the relationship properties
    query = "MATCH p=()-[r]-() "
    if rel_properties:
        query += " WHERE r.dummy_prop IS NULL "     # add a dummy statement for the loop to make sense
        for key, value in rel_properties.items():
            query += " AND r." + key + "='" + value + "' "
    query += " RETURN nodes(p), relationships(p), id(r)"

    subgraph = graph.run(query).data()
    # print(subgraph)
    # pprint(getmembers(subgraph))
    # print(subgraph[0]['nodes(p)'])
    # print(subgraph[0]['relationships(p)'])

    return subgraph


def extract_graph(graph, startnode=None, endnode=None, rel_properties=None):
    """
    See query_neo4j()
    """
    return query_neo4j(graph, startnode, endnode, rel_properties)


if __name__ == "__main__":
    main()

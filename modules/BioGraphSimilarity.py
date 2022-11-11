#
# 2021 Mikhail Kovalenko kovalenkom@health.missouri.edu
#
# 2022-11-01 TODO:
# comparison maps of values to match
# sets of properties to exclude
# figure out how to deal with aliases
# implement node-by-node retrieval and comparison instead of loading all into RAM
# switch to icecream debugger


# TODO switch away from py2neo to something better
# https://community.neo4j.com/t/are-there-any-real-instructions-for-using-py2neo/2689
# import py2neo
import re

# from inspect import getmembers
from pprint import pprint
from itertools import chain
import numpy as np
import pandas as pd
# from icecream import ic

from modules.EntityLookUpService import EntityLookUp

class GraphSimilarity:
    """ Class for comparing graphs in various biologically meaningful ways """

    # Compare nodes on the basis of this property
    # This is the default; overridable at class initialization
    node_property = "name"

    # A set of relationship types to exclude from comparison
    # 'alias' types are always excluded
    # Can be overriden by providing an empty set at initialization
    rel_ignore_set = {"alias"}

    # Scoring method
    # Default: "iou" Jaccard Similarity Score (Intersection Over Union)
    # Other choices:
    #    "overlap" Overlap coefficient      F = intersection(X, Y) / min(X, Y)
    #    "inclusion" Inclusion coefficient  F = intersection(X, Y) / length(Y)
    scoring_method = "iou"

    # Produce verbose diagnostic output
    debug = False

    def __init__(self, debug=False, match=None, ignore=None, score=None):
        """
        Initialize the class and all important variables

        :param debug: bool Print out debugging information
        :param match: string The property that nodes are compared by
        :param ignore: set The relationship types to ignore during comparisons
        :param score: string Use the given scoring method
        """
        self.debug = debug

        if match is not None:
            self.node_property = match

        if ignore is not None and type(ignore) is set:
            self.rel_ignore_set = ignore

        if score is not None:
            self.scoring_method = score

    def nodes(self, graph_ref, graph_new):
        """
        Compares only nodes of the given graphs using "Intersection over Union" (Jaccard Similarity) method.
        Uses node_property (or "name" by default) as the comparison point.
        Implemented with python's set methods union() and intersection().
        Returns a score (as a float) between 0 (no matches) and 1 (same set).
        NOTE: py2neo's graph.run(query).data() returns 'mutable sequences', not py2neo Subgraphs

        :param graph_ref: py2neo Subgraph or 'mutable sequence'
        :param graph_new: py2neo Subgraph or 'mutable sequence'
        :return: float between 0 and 1
        """

        if self.debug:
            print("> In " + self.__class__.__name__ + ".nodes:")

        # the input parameters may be 'mutable sequences', not py2neo Subgraphs
        nodes_ref = self.extract_nodes(graph_ref)
        nodes_new = self.extract_nodes(graph_new)

        # Extract the desired property from each node set into separate sets
        nodes_r = set()
        for nr in nodes_ref:
            nodes_r.add(dict(nr)[self.node_property])
        nodes_n = set()
        for nn in nodes_new:
            nodes_n.add(dict(nn)[self.node_property])

        intersection = nodes_r.intersection(nodes_n)
        nominator = len(intersection)
        union = nodes_r.union(nodes_n)

        if self.scoring_method == "overlap":
            denominator = min(len(nodes_ref), len(nodes_new))
        elif self.scoring_method == "inclusion":
            denominator = len(nodes_new)
        else:
            denominator = len(union)

        if not denominator:
            return 0

        score = nominator / denominator

        if self.debug:
            print(nodes_r)
            print(nodes_n)
            print(f"   Scoring: {self.scoring_method}")
            print(f"   Nodes formula: {nominator} / {denominator} = {score:+.2f}\n   --------\n")

        # compare to the alternative calculation if needed
        # score2 = self.nodes2(graph_ref, graph_new)
        # assert score == score2

        return float(score)

    def nodes2(self, graph_ref, graph_new):
        """
        Compares only nodes of the given graphs using "Intersection over Union" (Jaccard Similarity) method.
        Uses node_property (or "name" by default) as the comparison point.
        Implemented using for() loops.
        Returns a score (as a float) between 0 (no matches) and 1 (same set).
        NOTE: py2neo's graph.run(query).data() returns 'mutable sequences', not py2neo Subgraphs

        :param graph_ref: py2neo Subgraph or 'mutable sequence'
        :param graph_new: py2neo Subgraph or 'mutable sequence'
        :return: float between 0 and 1
        """

        if self.debug:
            print("> In " + self.__class__.__name__ + ".nodes2:")

        nodes_ref = self.extract_nodes(graph_ref)
        nodes_new = self.extract_nodes(graph_new)

        union = set()

        nominator = 0
        # loop over all nodes in each list
        for nr in nodes_ref:
            # use the node_property to collect all nodes into a union
            union.add(nr[self.node_property])

            for nn in nodes_new:
                # use the node_property to collect all nodes into a union
                union.add(nn[self.node_property])

                node_score = self._compare_nodes(nr, nn)
                nominator += node_score

        if self.scoring_method == "overlap":
            denominator = min(len(nodes_ref), len(nodes_new))
        elif self.scoring_method == "inclusion":
            denominator = len(nodes_new)
        else:
            denominator = len(union)

        if not denominator:
            return 0

        score = nominator / denominator

        if self.debug:
            print(union)
            print(f"   Scoring: {self.scoring_method}")
            print(f"   Nodes2 formula:  {nominator} / {denominator} = {score:+.2f}\n   --------\n")

        return float(score)

    def extract_nodes(self, graph):
        """
        Collects all nodes of a graph in an unordered unique set.
        Extraction is by Neo4j's node ID so there may be nodes with duplicate properties.

        :param graph: py2neo subgraph
        :return: set of node objects extracted from the graph
        """

        if self.debug:
            print(">>> In " + self.__class__.__name__ + ".extract_nodes:")

        nodes = set()
        ids = set()  # internal Neo4j IDs -- used here to help avoid duplicates

        if graph:
            for row in graph:
                for node in row['nodes(p)']:
                    # 'nodes(p)' returns a py2neo single-node Walkable py2neo.data.Subgraph here
                    #  that may contain duplicate records, especially if the relationship is non-directional
                    # if self.debug:
                    #     print(">>>>>> Node " + dict(node)['name'] + " breakdown:")
                    #     pprint(getmembers(node))

                    # python's set() is supposed to hold unique objects but
                    # it doesn't work with py2neo Node objects!
                    # therefore add nodes based on the node IDs
                    if node.identity not in ids:
                        nodes.add(node)
                        ids.add(node.identity)

        if self.debug:
            print("Neo4j IDs: " + str(ids))
            pprint(nodes)
        return nodes

    def _compare_nodes(self, node1, node2):
        """
        CompareS two nodes by the node_property, 'name' by default

        :param node1: Neo4j Node object
        :param node2: Neo4j Node object
        :return: float 0 or 1
        """

        # See if the nodes are proper py2neo.data.Node objects
        # Make sure we can compare them
        # assert type(node1) == py2neo.data.Node
        # assert type(node2) == py2neo.data.Node
        assert type(node1) == type(node2), "Don't know how to compare nodes of different data types"

        if dict(node1) and dict(node2):
            if dict(node1)[self.node_property] == dict(node2)[self.node_property]:
                if self.debug:
                    print(
                        f"-- {self.node_property}: {dict(node1)[self.node_property]} = "
                        "{dict(node2)[self.node_property]}: 1, ")
                return float(1)

        return float(0)

    def relationships(self, graph_ref, graph_new, directed=False):
        """
        Compares nodes and their immediate relationships of the given graphs.
        Returns a score (as a float) between 0 (no matches) and 1 (same node set with same relationships).

        :param graph_ref: py2neo Subgraph or 'mutable sequence'
        :param graph_new: py2neo Subgraph or 'mutable sequence'
        :param directed: flag to account for directionality, default value: False
        :return: float between 0 and 1
        """

        formula = ''  # to make PyCharm IDE happy
        if self.debug:
            print("> In " + self.__class__.__name__ + ".relationships:\t<< directed: " + str(directed))
            formula = "   Relationships Formula: ("

        rels_ref = self._extract_nodes_relations(graph_ref)
        rels_new = self._extract_nodes_relations(graph_new)

        # union = dict()  # dictionaries' keys are unique
        union = set()

        nominator = 0
        # loop over all nodes in each list
        for nr in rels_ref:
            # use the node_property to collect all nodes into a union
            union.add(nr['node'][self.node_property])

            for nn in rels_new:
                union.add(nn['node'][self.node_property])

                rel_score = self._compare_nodes_relations(nr, nn, directed)
                nominator += rel_score

                if rel_score:
                    if self.debug:
                        formula += f' {rel_score:+.2f}'

        if self.scoring_method == "overlap":
            denominator = min(len(rels_ref), len(rels_new))
        elif self.scoring_method == "inclusion":
            denominator = len(rels_new)
        else:
            denominator = len(union)
            # denominator = len(union.keys())

        if not denominator:
            return 0

        score = nominator / denominator

        if self.debug:
            formula += f" ) / {denominator} = {nominator} / {denominator} = {score:+.2f}"
            print(">>> Whole graph calculation: union")
            print(union)
            if directed:
                print(">>> Directed:")
            else:
                print(">>> Undirected:")
            print(f"{formula}\n   --------\n")

        return float(score)

    def _extract_nodes_relations(self, graph):
        """
        Count all relationships for each node of the given graph by type and return as a list of dicts.
        Only distinct neighbors are counted. Duplicate relationships to the same neighbor are counted as one.
        Output: list of dicts, each containing the Node object and subdicts 'in', 'out', and 'any'.
        Each subdict contains counts of each type of the node's relationships:
        [{'node': Node('N', name='CBF1', type='N'), 'in': {'act': 1, 'inh': 2}, 'out': {'act': 2}}, 'any": {}, {...}]

        :param graph: py2neo Subgraph or 'mutable sequence'
        :return: list of dicts containing the Node object and subdicts 'in', 'out', and 'any'
        """

        if self.debug:
            print(">> In " + self.__class__.__name__ + "._extract_nodes_relations:")

        # stores nodes and their relationship counts as a dict of dicts
        # NOTE: dictionaries' keys are unique
        relationship_counts = dict()

        processed_relations = set()
        graph_nodes = list()

        if graph:
            for row in graph:
                for rel in row['relationships(p)']:
                    # if self.debug:
                    #     print("Rel >>>>>> " + str(rel) + ":")
                    #     pprint(getmembers(rel))
                    # identity is the rel ID

                    # py2neo may return duplicate relationship records, so
                    # keep track of each relationship and do not process it twice
                    # using the ID assigned by Neo4j
                    if rel.identity in processed_relations:
                        continue

                    processed_relations.add(rel.identity)

                    if rel.relationships:
                        # get the type (name) of the relationship from 'relationships' tuple
                        rel_type = self._get_type(rel)

                        # skip those types that are excluded/ignored
                        # https://stackoverflow.com/questions/14829640/
                        for ign in self.rel_ignore_set:
                            if rel_type == ign:
                                break  # go back to the parent loop

                        # each relationship has a start node and an end node
                        start_name = rel.start_node['name']
                        end_name = rel.end_node['name']

                        # count directions separately
                        node_list = {'out': start_name, 'in': end_name}

                        for direction, node_name in node_list.items():
                            # create a dictionary for each node if it doesn't exist
                            if node_name not in relationship_counts:
                                relationship_counts[node_name] = dict()
                                if direction == 'out':
                                    relationship_counts[node_name]['node'] = rel.start_node
                                else:
                                    relationship_counts[node_name]['node'] = rel.end_node
                                # initialize collectors for directional counters
                                relationship_counts[node_name]['in'] = dict()
                                relationship_counts[node_name]['out'] = dict()

                                # Turns out that merging two dictionaries while adding up their values
                                #  is non-trivial in Python so this sub-dict will simplify things a lot
                                relationship_counts[node_name]['any'] = dict()

                            if rel_type not in relationship_counts[node_name][direction]:
                                relationship_counts[node_name][direction][rel_type] = set()
                            if rel_type not in relationship_counts[node_name]['any']:
                                relationship_counts[node_name]['any'][rel_type] = set()

                        # add the relationship names to collapse identical relationships
                        #  so we don't have eight (CYPA)-[ACTIVATES]->(MMP-9) links to throw our counts off
                        relationship_counts[start_name]['out'][rel_type].add(end_name)
                        relationship_counts[start_name]['any'][rel_type].add(end_name)
                        relationship_counts[end_name]['in'][rel_type].add(start_name)
                        relationship_counts[end_name]['any'][rel_type].add(start_name)

            # convert relationship names for each type of relationship into integer totals
            for k, v in relationship_counts.items():
                for d in ('in', 'out', 'any'):
                    for ky, vl in v[d].items():
                        v[d][ky] = len(vl)

            # strip the keys (node names) and return a list, not a dictionary
            for k, v in relationship_counts.items():
                # print("\t" + str(k) + ":\t" + str(v))
                graph_nodes.append(v)

            return graph_nodes

        # return an empty list if the graph is empty
        return []

    def _get_type(self, rel):
        """
        Obtains the type (name) of the relationship from 'relationships' tuple
        ('relationships', ({'source': 'PMC3365567', 'type': 'activate', 'vetted': 'yes'},))

        :param rel: py2neo 'relationships' tuple
        :return: string describing the relationship (hopefully uniquely)
        """

        # ##### WARNING: CRUDE WORKAROUNDS AHEAD! #####
        # If the 'relationships' record does not contain the 'type' attribute, we have a problem.
        # py2neo does not report a type of the relationship as its __name__ identifier (:LIKES)
        # but as the name of the relationship object (which should be Relationship instead),
        # so relationship's __name__ contains Neo4j's object ID instead (that has no obvious
        # way of being dereferenced into its type).

        if dict(rel.relationships[0]).get('type'):
            return dict(rel.relationships[0])['type']
        else:
            # try to extract the name of the relationship from the
            #  object type that Python reports
            rtype = re.search('\.data\.(.+?)\'>', str(type(rel)))
            if rtype:
                return rtype.group(1)
            else:
                # if no luck, use py2neo's identity parameter (often same as __name__)
                return str(rel.identity)

    def _compare_nodes_relations(self, node_relations1, node_relations2, directed=False):
        """
        Compares two nodes with their relationship counts by type.
        Returns 0 if the nodes don't match, a float between 0 and 1 if some match found.
        Input: list of dicts, each containing the Node object and subdicts 'in', 'out', and 'any'.
        Each subdict contains counts of each type of the node's relationships:
        [{'node': Node('N', name='CBF1', type='N'), 'in': {'act': 1, 'inh': 2}, 'out': {'act': 2}}, 'any": {}, {...}]

        :param node_relations1: list of dicts
        :param node_relations2: list of dicts
        :param directed: flag to account for directionality, default value: False
        :return: float between 0 and 1
        """

        if self.debug:
            print(">> In " + self.__class__.__name__ + "._compare_nodes_relations:\t<< directed: " + str(directed))
            # pprint(node_relations1['node'].identity)
            print(node_relations1)
            # pprint(node_relations2['node'].identity)
            print(node_relations2)

        # compare the nodes
        node_score = self._compare_nodes(node_relations1['node'], node_relations2['node'])
        if not node_score:
            return 0

        # Calculation:
        # If node comparison is unsuccessful, return 0 (above).
        # If the nodes are the same, add the ratios of each type of relationships to the node result (1), then
        # divide by the total number of types + the node result (1).
        # If undirected, use 'any' dict. If directed, use 'in' and 'out' dicts.

        # Undirected example: -->O-->   <--O-->   => Matches: nodes: 1, activate: 2/2=1
        #    => (1 + 1) / 2 = 1
        #    Nodes match with 2 'activate' relationships each
        # Directed example:   -->O-->   <--O-->   => Matches: nodes: 1, activate out: 1/2=0.5, activate in: 0/1=0
        #    => (1 + 0.5 + 0) / 3 = 0.5
        #    Nodes match halfway, score 0.5

        # Calculate the denominator -- all distinct rel types + 1 (node score)
        # Merge all distinct elements in 'in' and 'out' dicts for both entities being compared,
        # counting the keys and adding 1 to account for the node score.
        # The values are not important, only the keys.
        #
        # Denominator calculation has to be done HERE (as opposed to further below)
        #  because the loops below skip empty dicts entirely
        #  so the 0s they otherwise would have produces can't be counted
        if directed:
            union_in = {**node_relations1['in'], **node_relations2['in']}
            union_out = {**node_relations1['out'], **node_relations2['out']}
            denominator = len(union_in.keys()) + len(union_out.keys()) + 1
            if self.debug:
                print("UNION in:  " + str(len(union_in.keys())) + "\t" + str(union_in.keys()))
                print("UNION out: " + str(len(union_out.keys())) + "\t" + str(union_out.keys()))
        else:
            # union = {**node_relations1['in'], **node_relations2['in'], **node_relations1['out'],
            #          **node_relations2['out']}
            # denominator1 = len(union.keys()) + 1
            # if self.debug:
            #     print("UNION all: " + str(len(union.keys())) + "\t" + str(union.keys()))
            # or
            union = {**node_relations1['any'], **node_relations2['any']}
            denominator2 = len(union.keys()) + 1
            if self.debug:
                print("UNION any: " + str(len(union.keys())) + "\t" + str(union.keys()))
            # assert denominator1 == denominator2
            denominator = denominator2

        if self.debug:
            print("Denominator: " + str(denominator))

        # Calculate the nominator
        nominator = node_score
        formula = "   Formula: ( " + str(nominator)

        # directed: compare each relationship type separately from 'in' and 'out'
        # undirected: use 'any' where the types have already been added up, then compare totals for each node
        if directed:
            for dir in ['in', 'out']:
                # go through all types
                for rel1_type, count1 in node_relations1[dir].items():
                    for rel2_type, count2 in node_relations2[dir].items():
                        # compare the same types
                        if rel1_type == rel2_type:
                            sub_score = self._count_ratio(count1, count2)
                            nominator += sub_score

                            if self.debug:
                                print("-- Dir: " + dir + "; " + rel1_type + " score: " + str(sub_score))
                            formula += f' {sub_score:+.2f}'
        else:
            # go through all types
            for rel1_type, count1 in node_relations1['any'].items():
                for rel2_type, count2 in node_relations2['any'].items():
                    # compare the same types
                    if rel1_type == rel2_type:
                        sub_score = self._count_ratio(count1, count2)
                        nominator += sub_score

                        if self.debug:
                            print("-- Undir: " + rel1_type + " score: " + str(sub_score))
                        formula += f' {sub_score:+.2f}'

        score = nominator / denominator
        if self.debug:
            print(f"{formula} ) / {denominator} = {nominator:+.2f} / {denominator} = {score:+.2f}\n   --------")

        return float(score)

    def _count_ratio(self, count1, count2):
        """
        Returns the ratio of two numbers in 0..1 range.

        :param count1: one number
        :param count2: another number
        :return: float between 0 and 1
        """

        if not count1 or not count2:
            return 0

        ratio = count1 / count2
        if ratio > 1:
            ratio = 1 / ratio

        return float(ratio)

    def neighbors(self, graph_ref, graph_new, directed=False):
        """
        Compare nodes, relationships, and immediate neighbors of the given graphs.
        Returns a score (as a float) between 0 (no matches) and 1 (identical graphs).

        :param graph_ref: py2neo Subgraph or 'mutable sequence'
        :param graph_new: py2neo Subgraph or 'mutable sequence'
        :param directed: flag to account for directionality, default value: False
        :return: float between 0 and 1
        """

        if self.debug:
            print("> In " + self.__class__.__name__ + ".neighbors:\t<< directed: " + str(directed))

        neig_ref = self._extract_nodes_rels_neigs(graph_ref)
        neig_new = self._extract_nodes_rels_neigs(graph_new)

        union = dict()
        formula = "   Neighbors Formula: ("

        nominator = 0
        for nr in neig_ref:
            for nn in neig_new:
                # init keys for the denominator
                if not dict(nr['node'])[self.node_property] in union:
                    union[dict(nr['node'])[self.node_property]] = 0
                if not dict(nn['node'])[self.node_property] in union:
                    union[dict(nn['node'])[self.node_property]] = 0

                sub_score = self._compare_nodes_rels_neigs(nr, nn, directed)
                nominator += sub_score

                if sub_score:
                    # mark as compared for visual confirmation
                    union[dict(nr['node'])[self.node_property]] = sub_score
                    union[dict(nn['node'])[self.node_property]] = sub_score
                    if self.debug:
                        formula += f' {sub_score:+.2f}'

        if self.scoring_method == "overlap":
            denominator = min(len(neig_ref), len(neig_new))
        elif self.scoring_method == "inclusion":
            denominator = len(neig_new)
        else:
            # denominator is the union of all nodes from both graphs
            # denominator = len(union)
            denominator = len(union.keys())

        if not denominator:
            return 0

        score = nominator / denominator

        if self.debug:
            formula += f" ) / {denominator} = {nominator} / {denominator} = {score:+.2f}"
            print(">>> Whole graph calculation:")
            if directed:
                print(">>> Directed:")
            else:
                print(">>> Undirected:")
            print(union)
            print(f"{formula}\n   --------\n")

        return score

    def _extract_nodes_rels_neigs(self, graph):
        """
        Arrange all nodes into star clusters of node-centered immediate relationships and neighbors.
        Return as a list of dicts. Only distinct neighbors are counted.
        Duplicate relationships to the same neighbor are counted as one.
        Output: list of dicts, each containing the Node object and 'neighbors' subdicts.
        This particular format is chosen purely for the ease of later calculations.
        [{'node': Node('N', name='CBF1', type='N'), 'neighbors': [{'end': Node('N', name='CBF1', type='N'),
        relations: [{'type': 'activate', 'direction': 'out'}, {...}] }, {...}], {...}]

        :param graph: py2neo Subgraph or 'mutable sequence'
        :return: list of dicts containing the Node object, 'neighbors' dict, and 'relations' subdict
        """

        if self.debug:
            print(">> In " + self.__class__.__name__ + "._extract_nodes_rels_neigs:")

        # stores nodes and their immediate relationships and neighbors in a dict
        # NOTE: dictionaries' keys are unique!
        stars = dict()
        processed_relations = set()

        if graph:
            for row in graph:
                for rel in row['relationships(p)']:
                    if self.debug:
                        print("Rel >>>>>> ID: " + str(rel.identity) + " " + str(rel) + ":")
                        # pprint(getmembers(rel))
                    # identity is the rel ID

                    # keep track of each relationship and do not process it twice
                    # using the ID assigned by Neo4j
                    # py2neo returns duplicate relationship records
                    if rel.identity in processed_relations:
                        continue

                    processed_relations.add(rel.identity)

                    # if there's no relationship, there's nothing to compare
                    if rel.relationships:
                        # get the type (name) of the relationship from 'relationships' tuple
                        rel_type = self._get_type(rel)

                        # skip those types that are excluded/ignored
                        # https://stackoverflow.com/questions/14829640/
                        for ign in self.rel_ignore_set:
                            if rel_type == ign:
                                break  # go back to the parent loop

                        # create the dictionary for each node if it doesn't exist
                        # process both start and end nodes together because they come from Neo4j as a pair anyway
                        # so both directions will be collected

                        # use a unique node ID for tracking
                        # same graph, so no chance of a mix-up
                        start_node = rel.start_node.identity
                        end_node = rel.end_node.identity

                        # Exclude loops and self links from calculation - they don't carry information we want
                        if start_node == end_node:
                            continue

                        # Build both nodes at once
                        # direction follows from the location of the node in the relationship record
                        if start_node not in stars:
                            stars[start_node] = dict()
                            stars[start_node]['node'] = rel.start_node
                            # initialize collectors for directional counters
                            stars[start_node]['neighbors'] = dict()

                        if end_node not in stars:
                            stars[end_node] = dict()
                            stars[end_node]['node'] = rel.end_node
                            # initialize collectors for directional counters
                            stars[end_node]['neighbors'] = dict()

                        # Designate the mutual neighbor discarding duplicate entries
                        neighbor = {
                            'end': rel.end_node,
                            'relations': list()
                        }
                        if end_node not in stars[start_node]['neighbors']:
                            stars[start_node]['neighbors'][end_node] = neighbor

                        neighbor = {
                            'end': rel.start_node,
                            'relations': list()
                        }
                        if start_node not in stars[end_node]['neighbors']:
                            stars[end_node]['neighbors'][start_node] = neighbor

                        # Now collect all relationships
                        relation = {
                            'type': rel_type,
                            'direction': 'out',
                        }
                        if relation not in stars[start_node]['neighbors'][end_node]['relations']:
                            stars[start_node]['neighbors'][end_node]['relations'].append(relation)

                        relation = {
                            'type': rel_type,
                            'direction': 'in',
                        }
                        if relation not in stars[end_node]['neighbors'][start_node]['relations']:
                            stars[end_node]['neighbors'][start_node]['relations'].append(relation)

            # convert dictionaries into lists by stripping the keys
            stars = list(stars.values())
            for star in stars:
                star['neighbors'] = list(star['neighbors'].values())

            return stars

        # return an empty list if the graph is empty
        return []

    def _compare_nodes_rels_neigs(self, star1, star2, directed=False):
        """
        Compares two nodes including their relationships by type and direction and immediate neighbors.
        Duplicate relationships to the same end node are not counted, only presence/absense of a relationship
        Input format:
        [{'node': Node('N', name='CBF1', type='N'),
             'neighbors': [{'end': Node('N', name='CBF1', type='N'),
                     relations: [{'type': 'activate', 'direction': 'out'}, {...}] }, {...}], {...}]

        :param star1: list of dicts
        :param star2: list of dicts
        :param directed: flag to account for directionality, default value: False
        :return: float between 0 and 1
        """

        if self.debug:
            print(">> In " + self.__class__.__name__ + "._compare_nodes_rels_neigs:\t<< directed: " + str(directed))

        # compare the nodes
        node_score = self._compare_nodes(star1['node'], star2['node'])
        if not node_score:
            return 0

        # Calculate the denominator by adding all unique components of the comparison:
        #  the central node, the neighbor nodes, and their corresponding relationships
        union = dict()
        union[star1['node'][self.node_property]] = 1  # already matched

        if directed:
            for n in chain(star1['neighbors'], star2['neighbors']):
                union[n['end'][self.node_property]] = 0
                for r in n['relations']:
                    union[r['type'] + "_" + r['direction'] + "_" + n['end'][self.node_property]] = 0
        else:
            for n in chain(star1['neighbors'], star2['neighbors']):
                union[n['end'][self.node_property]] = 0
                for r in n['relations']:
                    union[r['type'] + "_" + n['end'][self.node_property]] = 0

        denominator = len(union)
        if not denominator:
            return 0

        if self.debug:
            pprint(star1)
            pprint(star2)
            pprint(union)
            print("- Denominator: " + str(denominator))

        # Calculation:
        #  for each node star, compare center nodes as usual,
        #  then loop over all neighbors and compare relationships for each matching neighbor
        #  1 if identical, 0 otherwise
        # We are deliberately not separating the comparisons of rel type and direction into two separate scores
        #  because a directed relationship is considered a single unit rather than separate parts.
        # Set the 'directed' flag to False to get that statistic.

        for n1 in star1['neighbors']:
            for n2 in star2['neighbors']:

                neighbor_score = self._compare_nodes(n1['end'], n2['end'])

                # Compare relations only for the matching nodes
                if neighbor_score:
                    union[n1['end'][self.node_property]] = neighbor_score

                    for r1 in n1['relations']:
                        for r2 in n2['relations']:
                            if r1['type'] == r2['type']:
                                if directed:
                                    if r1['direction'] == r2['direction']:
                                        rel = r1['type'] + "_" + r1['direction'] + "_" + n1['end'][self.node_property]
                                        if rel in union:
                                            union[rel] = 1
                                else:
                                    rel = r1['type'] + "_" + n1['end'][self.node_property]
                                    if rel in union:
                                        union[rel] = 1
        if self.debug:
            pprint(union)

        nominator = 0
        for v in union.values():
            nominator += v

        score = nominator / denominator

        if self.debug:
            print(f"   Star score: {nominator:+.2f} / {denominator} = {score:+.2f}\n   --------")

        return score

    def find_twins(self, graph_ref, graph_new, directed=False, tail=None, head=None, matrix=False):
        """
        Finds nodes that are similar in their relationships and neighbors.

        :param graph_ref: py2neo Subgraph or 'mutable sequence'
        :param graph_new: py2neo Subgraph or 'mutable sequence'
        :param directed: flag to account for directionality, default value: False
        :param tail: integer Only return this number of rows with lowest score
        :param head: integer Only return this number of rows with highest score
        :param matrix: bool True for output as a dataframe, False for a list sorted by the score
        :return:
        """

        if self.debug:
            print("> In " + self.__class__.__name__ + ".find_twins:")

        neig_ref = self._extract_nodes_rels_neigs(graph_ref)
        neig_new = self._extract_nodes_rels_neigs(graph_new)

        # dictionaries' keys are unique!
        scoreboard = dict()
        ssorted = dict()

        # threshold for neighborhoods of interest:
        # we want to see those that have more than one relation
        # so set the number of neighborhood elements (relations + neighbor nodes) here
        interesting = 3

        if not matrix:
            # the output is a dictionary sorted by the score
            for nr in neig_ref:
                for nn in neig_new:

                    sim_score = self._compare_twins(nr, nn, directed)

                    # omit nodes with the same node_property that match perfectly
                    # to cut down on redundant output
                    if sim_score == 1 and dict(nr['node'])[self.node_property] == dict(nn['node'])[self.node_property]:
                        pass
                    else:
                        # check for sim_score to weed out complete non-matches also
                        if sim_score and len(nn['neighbors']) >= interesting:
                            # https://stackoverflow.com/questions/23721230/float-values-as-dictionary-key
                            float_key = round(sim_score, 4)
                            if float_key not in scoreboard:
                                scoreboard[float_key] = set()

                            node_pair = dict(nr['node'])[self.node_property] + ":" + dict(nn['node'])[self.node_property]
                            scoreboard[float_key].add(node_pair)

            # sort by the score
            for key, value in sorted(scoreboard.items(), reverse=True):
                ssorted[key] = value

            output = ssorted
            if tail:
                output = dict(list(ssorted.items())[-tail:])
            if head:
                output = dict(list(ssorted.items())[:head])

        else:
            # the output is a matrix (2D array)
            x = len(neig_ref)
            y = len(neig_new)
            matrix = np.zeros([x, y], dtype=np.float)
            row_labels = list()
            col_labels = list()

            # init labels
            for nn in neig_new:
                col_labels.append(nn['node'][self.node_property])

            for nr in neig_ref:
                # init more labels
                row_labels.append(nr['node'][self.node_property])
                for nn in neig_new:
                    sim_score = self._compare_twins(nr, nn, directed)
                    matrix[row_labels.index(nr['node'][self.node_property])][
                        col_labels.index(nn['node'][self.node_property])] = sim_score
                    pass

            df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

            df.sort_index(inplace=True)
            df.sort_index(axis=1, inplace=True)
            # df.to_csv("output/twins.csv")

            # remove rows with all zeros
            df = df.loc[(df != 0).any(axis=1)]
            # remove columns with all zeros
            df = df.loc[:, (df != 0).any(axis=0)]

            output = df

        return output

    def _compare_twins(self, star1, star2, directed=False):
        """
        Compares two star nodes without scoring the center node.
        A copy of self._compare_nodes_rels_neigs() without comparing the central nodes.

        :param star1: list of dicts
        :param star2: list of dicts
        :param directed: flag to account for directionality, default value: False
        :return: float between 0 and 1
        """

        if self.debug:
            print(">> In " + self.__class__.__name__ + "._compare_twins:")

        # Calculate the denominator by adding all unique components of the comparison:
        #  the neighbor nodes, and their corresponding relationships, but not the central node
        union = dict()

        if directed:
            for n in chain(star1['neighbors'], star2['neighbors']):
                union[n['end'][self.node_property]] = 0
                for r in n['relations']:
                    union[r['type'] + "_" + r['direction'] + "_" + str(n['end'][self.node_property])] = 0
        else:
            for n in chain(star1['neighbors'], star2['neighbors']):
                union[n['end'][self.node_property]] = 0
                for r in n['relations']:
                    union[r['type'] + "_" + n['end'][self.node_property]] = 0

        denominator = len(union)
        if not denominator:
            return 0

        if self.debug:
            pprint(star1)
            pprint(star2)
            pprint(union)
            print("- Denominator: " + str(denominator))

        # Calculation:
        #  for each node star, loop over all neighbors and compare relationships for each matching neighbor
        #  1 if identical, 0 otherwise
        # We are deliberately not separating the comparisons of rel type and direction into two separate scores
        #  because a directed relationship is considered a single unit rather than separate parts.
        # Set the 'directed' flag to False to get that statistic.

        for n1 in star1['neighbors']:
            for n2 in star2['neighbors']:

                neighbor_score = self._compare_nodes(n1['end'], n2['end'])

                # Compare relations only for the matching nodes
                if neighbor_score:
                    union[n1['end'][self.node_property]] = neighbor_score

                    for r1 in n1['relations']:
                        for r2 in n2['relations']:
                            if r1['type'] == r2['type']:
                                if directed:
                                    if r1['direction'] == r2['direction']:
                                        rel = r1['type'] + "_" + r1['direction'] + "_" + str(
                                            n1['end'][self.node_property])
                                        if rel in union:
                                            union[rel] = 1
                                else:
                                    rel = r1['type'] + "_" + n1['end'][self.node_property]
                                    if rel in union:
                                        union[rel] = 1
        if self.debug:
            pprint(union)

        nominator = 0
        for v in union.values():
            nominator += v

        score = nominator / denominator

        if self.debug:
            print(f"   Star score: {nominator:+.2f} / {denominator} = {score:+.2f}\n   --------")

        return score

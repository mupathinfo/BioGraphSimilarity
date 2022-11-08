# BioGraphSimilarity

## Introduction

This class offers a tentative implementation of the methods described in AMIA 2022 Annual Symposium poster publication Exploration of Disease Pathways Using Simple Graph Comparison Methods.

Biological pathway graphs require comparison methods that focus on specific features of interest, such as the type of relationships and their directionality, which are often ignored in common graph database. A biological pathway graph is semantically richer than a simple vertice-edge schematic. To capture the richness of recorded biological interactions, we calculate the similarity of each node including its relationships before calculating the overall similarity of the whole graph.

## Methods

### `nodes(), nodes2()`

Compare only the nodes for a quick assessment of two graphs using certain node-specific criteria, such as the name or a database ID. When aggregated, this score offers a quick estimate at how close two pathways are in terms of their molecular elements.

### `relationships()`

Considers each node together with its immediate relationships as a comparison unit. Relationships can be compared by their type (activate, inhibit, etc.) and by direction. Direction of action can be ignored for the purposes of a less strict match. Immediate neighbor nodes are not compared. This score allows to compare biological pathway elements’ major actions and effects.

### `neighbors()`

The strictest measure of similarity of semantically rich graphs in terms of all material biological pathway components is given by comparing nodes together with their adjacent neighbors and their direct relationships. Multiple relationships between a node and each neighbor are counted separately, while the neighbor itself is accounted for only once.

Each level of complexity contributes additional information, lowering the aggregate similarity score with every dissimilar detail. Once all node-to-node similarity scores are calculated, they are aggregated using Jaccard similarity (Intersection Over Union), Overlap, or Inclusion coefficients.

### `find_twins()`

Measuring neighbor similarity while disregarding central nodes allows us to search for similar graph elements with different names that may warrant closer attention from the investigator. In practice, this is useful when comparing pathways originating from different sources, where some node descriptions may differ slightly.

Two nodes with the same twin similarity score are alike in some way and may exhibit similar properties because they interact with the same entities in the similar manner, even though the nodes themselves don’t match based on their compared internal properties (such as the name or ID).

## Authors

* Mikhail Kovalenko, MS
* Mihail Popescu, PhD

University of Missouri-Columbia

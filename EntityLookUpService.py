#
# 2021 Mikhail Kovalenko kovalenkom@health.missouri.edu
#

class EntityLookUp:
    """
    A dummy implementation of a node/relationship lookup service
    To be implemented as the need arises

    This is intended for reconciling the variety of names extracted
    from heterogeneous sources for the purposes of comparing canonical
    gene/protein names and eliminating aliases from comparisons
    """

    dummy = True

    def __init__(self, service=None):
        if service == 'sqlite':
            self.dummy = False
            self.__sqlite()
        elif service == 'mysql':
            self.dummy = False
            self.__mysql()
        elif service == 'online':
            self.dummy = False
            self.__online()
        else:
            self.dummy = True

    def find_by_name(self, name):
        """
        Look up a node by its "name" property and return a likely match
        or None if no matches found

        :param name: string     Name of the node
        :return: string or None Canonical name of the entity
        """
        if self.dummy is True:
            return name.strip()

    def find_by_id(self, node_id, database=None):
        """
        Look up a node by its ID optionally specifying a database to use
        and return a likely name match or None if no matches found

        :param node_id: string  Database ID
        :param database: string Specific database to use, or all known if None given
        :return: string or None Canonical name of the entity
        """
        if self.dummy is True:
            return node_id.strip()

    def match_relation(self, rel_type):
        """
        Look up a type of an relationship (edge) and return a likely match

        :param rel_type:  string Relationship type
        :return: string or None
        """
        if self.dummy is True:
            return rel_type.strip()

    def __sqlite(self):
        """
        Initiate a connection to an SQLite database
        :return: connection object
        """
        pass

    def __mysql(self):
        """
        Initiate a connection to a MySQL database
        :return: connection object
        """
        pass

    def __online(self):
        """
        Initiate a connection to an online resource
        :return: connection object
        """
        pass

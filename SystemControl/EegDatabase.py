"""
@title
    EegDatabase
@desc
    SQLLite database for storing and querying recorded eeg data
"""
import argparse
import sqlite3
from sqlite3 import Error

from SystemControl import DATABASE_URL


class EegDatabase:

    def __init__(self, db_path: str = DATABASE_URL):
        self.db_path = db_path
        self.__db = None
        return

    def __execute_command(self, command_str: str):
        db_cursor = self.__db.cursor()
        db_cursor.execute(command_str)
        self.__db.commit()
        return

    def connect(self):
        """
        Create a database connection to a SQLite database.

        :return:
        """
        try:
            self.__db = sqlite3.connect(self.db_path)
            print('SQLLite version: {}'.format(sqlite3.version))
        except Error as e:
            print(str(e))
        return self.__db

    def close(self):
        """

        :return:
        """
        if self.__db:
            self.__db.close()
        return

    def create_table(self, table_name: str, table_columns: dict) -> bool:
        field_str = ''
        field_sep = ''
        for each_col_name, each_type in table_columns.items():
            field_str += '{}{} {}'.format(field_sep, each_type, each_col_name)
            field_sep = ', '

        create_command = 'create table if not exists {}({})'.format(table_name, field_str)
        self.__execute_command(create_command)
        tbl_exists = self.check_table_exists(table_name)
        return tbl_exists

    def delete_table(self, table_name: str) -> bool:
        delete_command = 'drop table if exists {}'.format(table_name)
        self.__execute_command(delete_command)
        tbl_exists = self.check_table_exists(table_name)
        return tbl_exists

    def check_table_exists(self, table_name: str) -> bool:
        """

        :param table_name:
        :return:
        """
        table_names = self.get_tables()
        table_exists = False
        if table_name in table_names:
            table_exists = True
        return table_exists

    # noinspection SqlResolve
    def get_tables(self) -> list:
        """
        To list all tables in a SQLite3 database, you should query sqlite_master table and then use
        the fetchall() to fetch the results from the SELECT statement.

        The sqlite_master is the master table in SQLite3 which stores all tables.
        :return:
        """
        db_cursor = self.__db.cursor()
        db_cursor.execute('SELECT name from sqlite_master where type= "table"')
        table_names = [
            each_entry[0]
            for each_entry in db_cursor.fetchall()
        ]
        return table_names


def main(args):
    """

    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    test_table_name = 'test_table'
    test_table_fields = {
        'id': 'int',
        'name': 'str'
    }

    eeg_db = EegDatabase()
    eeg_db.connect()
    tbls = eeg_db.get_tables()
    tbl_exists = eeg_db.check_table_exists(test_table_name)
    print('table names:\n\t{}'.format('\n\t'.join(tbls)))
    print('table exists:\n\t{} -> {}'.format(test_table_name, tbl_exists))
    eeg_db.create_table(test_table_name, test_table_fields)
    tbls = eeg_db.get_tables()
    tbl_exists = eeg_db.check_table_exists(test_table_name)
    print('table names:\n\t{}'.format('\n\t'.join(tbls)))
    print('table exists:\n\t{} -> {}'.format(test_table_name, tbl_exists))
    eeg_db.delete_table(test_table_name)
    tbls = eeg_db.get_tables()
    tbl_exists = eeg_db.check_table_exists(test_table_name)
    print('table names:\n\t{}'.format('\n\t'.join(tbls)))
    print('table exists:\n\t{} -> {}'.format(test_table_name, tbl_exists))
    eeg_db.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')

    cargs = parser.parse_args()
    main(cargs)

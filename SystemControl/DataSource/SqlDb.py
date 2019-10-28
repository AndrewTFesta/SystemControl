"""
@title
    EegDatabase
@desc
    SQLLite database for storing and querying recorded eeg data
"""
import argparse
import sqlite3
from enum import Enum, auto
from sqlite3 import Error

from SystemControl import DATABASE_URL


class DataType(Enum):
    NULL = auto()
    INTEGER = auto()
    REAL = auto()
    TEXT = auto()
    BLOB = auto()


class SqlDb:
    # todo  add checks for is_connected before attempting most functionality

    def __init__(self, db_path: str = DATABASE_URL):
        self.db_path = db_path
        self.__db = None
        return

    def is_connected(self):
        if isinstance(self.__db, sqlite3.Connection):
            conn = True
        else:
            conn = False
        return conn

    def connect(self):
        """
        Create a database connection to a SQLite database.

        :return:
        """
        if not self.is_connected():
            try:
                self.__db = sqlite3.connect(self.db_path)
                sqlite3.enable_callback_tracebacks(True)
                print('SQLLite version: {}'.format(sqlite3.version))
            except Error as e:
                print(str(e))
        else:
            print(f'Already connected to sqlite database {self.__db}')
        return self.__db

    def close(self):
        """

        :return:
        """
        if self.__db:
            self.__db.close()
        return

    def create_table(self, table_name: str, table_columns: dict) -> bool:
        # todo  add ability to set constraints on specific fields
        #       field name -> {type:str, constraints:list}
        #       multiple fields cannot be primary key
        # todo  add ability to create superkey
        field_str = 'id INTEGER PRIMARY KEY'
        field_sep = ', '
        for each_col_name, each_type in table_columns.items():
            field_str += f'{field_sep}{each_col_name} {each_type}'

        create_command = f'create table if not exists {table_name}({field_str})'

        self.__db.execute(create_command)
        self.__db.commit()

        tbl_exists = self.check_table_exists(table_name)
        return tbl_exists

    def delete_table(self, table_name: str) -> bool:
        delete_command = 'drop table if exists {}'.format(table_name)

        self.__db.execute(delete_command)
        self.__db.commit()

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
        get_tables_command = 'SELECT name from sqlite_master where type= "table"'

        db_cursor = self.__db.cursor()
        db_cursor.execute(get_tables_command)
        # self.__db.commit()

        table_names = [
            each_entry[0]
            for each_entry in db_cursor.fetchall()
        ]
        db_cursor.close()
        return table_names

    def insert(self, table_name: str, data_dict: dict):
        table_vals = list(data_dict.values())

        table_fields_str = ','.join(list(data_dict.keys()))
        table_vals_str = ','.join(['?'] * len(data_dict.keys()))

        insert_command = f'insert into {table_name}({table_fields_str}) values({table_vals_str})'

        self.__db.execute(insert_command, table_vals)
        self.__db.commit()
        return

    def insert_multiple(self, table_name: str, data_list: list):
        val_list = [list(each_entry.values()) for each_entry in data_list]
        last_entry = data_list[-1]

        table_fields_str = ','.join(list(last_entry.keys()))
        table_vals_str = ','.join(['?'] * len(last_entry.keys()))

        insert_command = f'insert into {table_name}({table_fields_str}) values({table_vals_str})'

        self.__db.executemany(insert_command, val_list)
        self.__db.commit()
        return

    def get_table_rows(self, table_name, filter_dict=None) -> list:
        select_command = f'select * from {table_name}'
        if filter_dict:
            select_command += ' where'
            sep = ''
            for each_key, each_val in filter_dict.items():
                select_command += f' {sep}{each_key}=:{each_key}'
                sep = 'and '
        db_cursor = self.__db.cursor()
        db_cursor.execute(select_command, filter_dict)
        # self.__db.commit()

        row_list = db_cursor.fetchall()
        return row_list

    def load_data(self):
        # TODO
        raise NotImplementedError


def main(args):
    """

    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    test_table_name = 'test_table'
    test_table_fields = {
        'name': DataType.TEXT
    }
    db_path = DATABASE_URL
    eeg_db = SqlDb(db_path)
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

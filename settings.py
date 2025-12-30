# @FileName: settings

class BaseConfig(object):
    HOST = '0.0.0.0'
    PORT = 3333
    # SQLALCHEMY_DATABASE_URI = {'a':'1'}
    # SQLALCHEMY_BINDS = {'b':'2'}
    pass


class DevelopmentConfig(BaseConfig):
    pass


class TestConfig(BaseConfig):
    pass


class ProductionConfig(BaseConfig):
    pass
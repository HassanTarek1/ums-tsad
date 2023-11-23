
import settings
# from extensions import db
from controller.mdata.mdata import mdata_bp
from controller.mevaluation.mevaluation import mevaluation_bp
from controller.mmodel.mmodel import mmodel_bp
from flask import Flask


def create_app():

    # load config
    app = Flask(__name__)
    app.config.from_object(settings.DevelopmentConfig)

    # register blueprint
    app.register_blueprint(mdata_bp)
    app.register_blueprint(mevaluation_bp)
    app.register_blueprint(mmodel_bp)

    # # init extensions
    # db.init_app(app)

    return app
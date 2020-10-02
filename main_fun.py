import logging
def main_fun(argv, ctx):
    logging.info('test test')

    import model
    cluster_spec, server = TFNode.start_cluster_server(ctx)
    #model.main(cluster_spec, server)
    logging.info('test test')
    exit(1)
    model.main()

if __name__ == '__main__':
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from tensorflowonspark import TFCluster, TFNode
    import argparse

    sc = SparkContext(conf=SparkConf().setAppName("RUL"))

    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", help="number of nodes in the cluster (for Spark Standalone)", type=int,
                        default=num_executors)
    parser.add_argument("--num_ps", help="number of parameter servers", type=int, default=1)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
    args = parser.parse_args()
    print(args.cluster_size, args.num_ps)
    #cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)

    cluster.shutdown()

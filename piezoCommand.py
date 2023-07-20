import argparse
import sys


def main():	
	parser = argparse.ArgumentParser(add_help=False)
	#受け取る引数を追加
	commandName = sys.argv[1]
	parser.add_argument("commandName")
	
	#oci-stop-instanceコマンドの時
	if commandName == "":
		parser.add_argument("--force-stop", required=True)
		parser.add_argument("--get-state", action="store_true")
		args = parser.parse_args()
		
		#強制停止コマンド
		#STOPの場合
		if args.force_stop == "y":
			try:
				instanceData = instance.StopInstance(args.instanceID, is_force_stop=True)
			except Exception as e:
				#例外処理
				errorHandling = ociError.OciErrorHandling()
				errorHandling.SetError(e)
				errorHandling.Execute()			

		#SOFTSTOPの場合
		elif args.force_stop == "n":
			try:
				instanceData = instance.StopInstance(args.instanceID, is_force_stop=False)
			except Exception as e:
				#例外処理
				errorHandling = ociError.OciErrorHandling()
				errorHandling.SetError(e)
				errorHandling.Execute()	
		
		
		print("Request successfully processed.")
		
		#--get-stateオプションが選択された時、インスタンスの情報を表示する。
		if args.get_state:			
			print(f"instanceName: {instanceData.display_name}")
			print(f"instanceState: {instanceData.lifecycle_state}")
	
	#oci-get-instanceコマンドの時
	elif commandName == "oci-instance-get-state":
		parser.add_argument("--stop-info", action="store_true")
		args = parser.parse_args()

		try:
			instanceData = instance.GetInstanceState(args.instanceID)
		except Exception as e:
			#例外処理
			errorHandling = ociError.OciErrorHandling()
			errorHandling.SetError(e)
			errorHandling.Execute()			
		
		#--stop-infoオプションが指定された時
		if args.stop_info:
			if instanceData.lifecycle_state == "STOPPED":
				print("yes")
			else:
				print("no")
		else:
			print(f"instanceName: {instanceData.display_name}")
			print(f"instanceState: {instanceData.lifecycle_state}")

			
if __name__ == "__main__":
    main()
